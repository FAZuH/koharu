// The wgpu type graph behind `Rasterizer` (now held by `ServerState`) overflows
// the default auto-trait recursion depth during `Send` evaluation (rust-lang#159228).
#![recursion_limit = "256"]

use std::{
    collections::{BTreeMap, VecDeque},
    future::Future,
    io,
    net::{IpAddr, Ipv4Addr},
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex as StdMutex},
    time::{Duration, Instant},
};

use anyhow::Result;
use axum::{
    Json, Router,
    extract::{FromRequest, Path, Request, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use clap::Parser;
use image::{
    ExtendedColorType, ImageEncoder as _, ImageFormat, ImageReader,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use koharu_config::Config;
use koharu_ml::Device;
use koharu_pipeline::{
    Committer, Operation, Pipeline, PipelineConfig, Progress, ProgressSink, Report,
    Request as PipelineRequest, RunStatus, Scope, Stage, StageOutput, StopToken,
};
use koharu_psd::{PsdExportOptions, export_page};
use koharu_rasterizer::{RasterOptions, Rasterizer};
use koharu_renderer::Renderer;
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, EntityId, PageDraft, Session, Snapshot,
};
use koharu_translator::{Language, ProvidersConfig};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::{
    net::TcpListener,
    sync::{Mutex, OnceCell},
};

const DEFAULT_PORT: u16 = 4100;
const PROGRESS_RING_CAPACITY: usize = 32;

#[derive(Debug, Parser)]
#[command(version, about = "Koharu headless HTTP and MCP server")]
struct Arguments {
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    host: IpAddr,

    #[arg(long, value_name = "PORT", default_value_t = DEFAULT_PORT)]
    port: u16,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    debug: bool,
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
    version: &'static str,
}

#[derive(Serialize)]
struct ApiError {
    status: u16,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status: status.as_u16(),
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(self)).into_response()
    }
}

/// JSON body extractor whose rejections keep the `{"status","message"}`
/// envelope instead of axum's default plain-text responses.
struct JsonBody<T>(T);

impl<T, S> FromRequest<S> for JsonBody<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        Json::<T>::from_request(request, state)
            .await
            .map(|Json(value)| Self(value))
            .map_err(|rejection| ApiError::new(rejection.status(), rejection.body_text()))
    }
}

struct OpenProject {
    session: Session,
    path: PathBuf,
    name: String,
}

type RunFuture = Pin<Box<dyn Future<Output = Result<Report>> + Send>>;

/// Seam between the HTTP job registry and a pipeline run. Production wires
/// `Pipeline::execute` here exactly once; tests inject a runner that needs
/// no model weights. The pipeline is owned by `ServerState` so every run
/// shares one serial execution guard (D7) and one config watcher.
#[derive(Clone)]
struct Runner(
    Arc<dyn Fn(Pipeline, Snapshot, PipelineRequest, ProjectCommitter) -> RunFuture + Send + Sync>,
);

impl Runner {
    fn execute() -> Self {
        Self(Arc::new(|pipeline, snapshot, request, mut committer| {
            Box::pin(async move { Ok(pipeline.execute(snapshot, request, &mut committer).await?) })
        }))
    }
}

struct ProjectCommitter {
    project: Arc<Mutex<Option<OpenProject>>>,
}

#[async_trait::async_trait]
impl Committer for ProjectCommitter {
    async fn commit(&mut self, output: StageOutput) -> Result<Snapshot> {
        let mut current = self.project.lock().await;
        let Some(project) = current.as_mut() else {
            anyhow::bail!("project was closed while the pipeline was running");
        };
        Ok(project.session.commit(output.patch).await?.snapshot)
    }
}

#[derive(Clone, Serialize)]
#[serde(
    tag = "event",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
enum ProgressView {
    Started {
        pages: Vec<String>,
        stages: Vec<String>,
    },
    Loading {
        page: String,
        stage: String,
        model: String,
    },
    Running {
        page: String,
        stage: String,
        model: String,
    },
    Finished {
        page: String,
        stage: String,
        model: String,
        elapsed_ms: u64,
    },
    Skipped {
        page: String,
        stage: String,
    },
}

impl From<Progress> for ProgressView {
    fn from(event: Progress) -> Self {
        match event {
            Progress::Started { pages, stages } => Self::Started {
                pages: pages.into_iter().map(|page| page.to_string()).collect(),
                stages: stages.into_iter().map(|stage| stage.to_string()).collect(),
            },
            Progress::Loading { page, stage, model } => Self::Loading {
                page: page.to_string(),
                stage: stage.to_string(),
                model,
            },
            Progress::Running { page, stage, model } => Self::Running {
                page: page.to_string(),
                stage: stage.to_string(),
                model,
            },
            Progress::Finished {
                page,
                stage,
                model,
                elapsed,
            } => Self::Finished {
                page: page.to_string(),
                stage: stage.to_string(),
                model,
                elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
            },
            Progress::Skipped { page, stage } => Self::Skipped {
                page: page.to_string(),
                stage: stage.to_string(),
            },
        }
    }
}

fn push_progress(progress: &StdMutex<VecDeque<ProgressView>>, event: Progress) {
    let mut ring = progress.lock().expect("progress ring is poisoned");
    if ring.len() == PROGRESS_RING_CAPACITY {
        ring.pop_front();
    }
    ring.push_back(ProgressView::from(event));
}

#[derive(Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum JobStatus {
    Running,
    Completed,
    Stopped,
    Failed,
}

struct Job {
    id: String,
    status: JobStatus,
    message: Option<String>,
    completed: usize,
    total: usize,
    elapsed: Option<Duration>,
    started: Instant,
    progress: Arc<StdMutex<VecDeque<ProgressView>>>,
    stop: StopToken,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OperationView {
    id: String,
    status: JobStatus,
    completed: usize,
    total: usize,
    elapsed: f64,
    recent_progress: Vec<ProgressView>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

/// Finished jobs are kept for the lifetime of the process so a script can poll
/// an operation it started long ago.
/// ponytail: unbounded over the runs of one process (restarts clear it); cap
/// or evict terminal jobs past an age if a long-lived server ever needs it.
#[derive(Default)]
struct Operations(StdMutex<Vec<Job>>);

impl Operations {
    fn start(&self, job: Job) -> bool {
        let mut jobs = self.0.lock().expect("operations registry is poisoned");
        if jobs.iter().any(|job| job.status == JobStatus::Running) {
            return false;
        }
        jobs.push(job);
        true
    }

    fn is_running(&self) -> bool {
        self.0
            .lock()
            .expect("operations registry is poisoned")
            .iter()
            .any(|job| job.status == JobStatus::Running)
    }

    fn stop(&self, id: &str) {
        let mut jobs = self.0.lock().expect("operations registry is poisoned");
        if let Some(job) = jobs.iter_mut().find(|job| job.id == id)
            && job.status == JobStatus::Running
        {
            job.stop.stop();
        }
    }

    fn finish(&self, id: &str, outcome: Result<Report>) {
        let mut jobs = self.0.lock().expect("operations registry is poisoned");
        let Some(job) = jobs.iter_mut().find(|job| job.id == id) else {
            return;
        };
        match outcome {
            Ok(report) => {
                job.status = match report.status {
                    RunStatus::Completed => JobStatus::Completed,
                    RunStatus::Stopped => JobStatus::Stopped,
                };
                job.completed = report.completed;
                job.total = report.total;
                job.elapsed = Some(report.elapsed);
            }
            Err(error) => {
                job.status = JobStatus::Failed;
                job.message = Some(error.to_string());
                job.elapsed = Some(job.started.elapsed());
            }
        }
    }

    fn view(&self) -> Vec<OperationView> {
        let jobs = self.0.lock().expect("operations registry is poisoned");
        jobs.iter()
            .map(|job| OperationView {
                id: job.id.clone(),
                status: job.status,
                completed: job.completed,
                total: job.total,
                elapsed: job
                    .elapsed
                    .unwrap_or_else(|| job.started.elapsed())
                    .as_secs_f64(),
                recent_progress: job
                    .progress
                    .lock()
                    .expect("progress ring is poisoned")
                    .iter()
                    .cloned()
                    .collect(),
                message: job.message.clone(),
            })
            .collect()
    }
}

/// Translation overrides that select a run-scoped pipeline.
type OverrideKey = (Option<Language>, Option<String>);

/// Returns the memoized pipeline when its key matches `key`, so an identical
/// override set reuses one pipeline instead of building another.
fn memo_hit(memo: &Option<(OverrideKey, Pipeline)>, key: &OverrideKey) -> Option<Pipeline> {
    let (cached, pipeline) = memo.as_ref()?;
    (cached == key).then(|| pipeline.clone())
}

struct ServerState {
    project: Arc<Mutex<Option<OpenProject>>>,
    operations: Operations,
    pipeline: Pipeline,
    device: Device,
    /// Memo of the last override-built pipeline. `Pipeline::from_config` spawns
    /// a config-watcher task that never terminates, so runs that override the
    /// translation share this single slot instead of building one per run.
    /// ponytail: one parked watcher per memo miss (repeated identical keys
    /// reuse the slot; alternating keys leak per run); the real fix — a
    /// run-scoped translation override on `koharu_pipeline::Request` or a
    /// watcher-free constructor — is an upstream proposal.
    overridden: StdMutex<Option<(OverrideKey, Pipeline)>>,
    export: OnceCell<ExportEngine>,
    runner: Runner,
}

impl ServerState {
    fn new(pipeline: Pipeline, device: Device, runner: Runner) -> Self {
        Self {
            project: Arc::default(),
            operations: Operations::default(),
            pipeline,
            device,
            overridden: StdMutex::new(None),
            export: OnceCell::new(),
            runner,
        }
    }

    /// Runs without request-level overrides reuse the pipeline built once at
    /// startup from the on-disk config (D14); an overridden run builds a
    /// one-off pipeline from the same config so the user's model and provider
    /// choices survive, with only the request's translation applied.
    fn pipeline_for(
        &self,
        target_language: Option<Language>,
        system_prompt: Option<&str>,
    ) -> Result<Pipeline> {
        let key = (target_language, system_prompt.map(str::to_owned));
        if key.0.is_none() && key.1.is_none() {
            return Ok(self.pipeline.clone());
        }
        if let Some(pipeline) = memo_hit(
            &self
                .overridden
                .lock()
                .expect("overridden pipeline memo is poisoned"),
            &key,
        ) {
            return Ok(pipeline);
        }
        let mut config = PipelineConfig::load()?.read()?.clone();
        if let Some(target_language) = target_language {
            config.translation.target_language = target_language;
        }
        if let Some(system_prompt) = system_prompt {
            config.translation.instructions = Some(system_prompt.to_owned());
        }
        let pipeline = Pipeline::from_config(
            Config::memory(config),
            ProvidersConfig::load()?,
            self.device.clone(),
        )?;
        *self
            .overridden
            .lock()
            .expect("overridden pipeline memo is poisoned") = Some((key, pipeline.clone()));
        Ok(pipeline)
    }
}

type AppState = Arc<ServerState>;

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();

    if arguments.debug {
        tracing_subscriber::fmt::init();
    }

    if !arguments.host.is_loopback() {
        eprintln!(
            "warning: binding {} exposes the unauthenticated API beyond loopback",
            arguments.host
        );
    }

    let listener = bind(&arguments.host, arguments.port).await?;

    koharu_ml::init().await?;
    let device = koharu_ml::device(arguments.cpu);
    let pipeline = Pipeline::load(device.clone())?;

    println!(
        "koharu-rpc listening on http://{}:{}",
        arguments.host, arguments.port
    );

    axum::serve(
        listener,
        router(Arc::new(ServerState::new(
            pipeline,
            device,
            Runner::execute(),
        ))),
    )
    .with_graceful_shutdown(async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install ctrl-c handler");
    })
    .await?;

    Ok(())
}

async fn bind(host: &IpAddr, port: u16) -> io::Result<TcpListener> {
    TcpListener::bind((*host, port)).await.map_err(|err| {
        if err.kind() == io::ErrorKind::AddrInUse {
            io::Error::new(
                err.kind(),
                format!("port {port} is busy on {host}; pass --port"),
            )
        } else {
            err
        }
    })
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/project", get(get_project))
        .route("/api/v1/project/open", post(open_project))
        .route("/api/v1/project/pages", post(add_pages))
        .route("/api/v1/pipelines", post(start_pipeline))
        .route("/api/v1/operations", get(list_operations))
        .route("/api/v1/operations/{id}", delete(stop_operation))
        .route("/api/v1/pages/{id}/image", get(page_image))
        .route("/api/v1/export", post(export))
        .fallback(not_found)
        .with_state(state)
}

async fn health() -> Json<Health> {
    Json(Health {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

async fn not_found(request: Request) -> Response {
    ApiError::new(
        StatusCode::NOT_FOUND,
        format!("no route for {}", request.uri().path()),
    )
    .into_response()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PageView {
    id: String,
    index: usize,
    width: f64,
    height: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProjectView {
    name: String,
    path: String,
    pages: Vec<PageView>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenProjectRequest {
    path: String,
    create: Option<bool>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AddPagesRequest {
    paths: Vec<String>,
    index: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AddPagesResponse {
    pages: Vec<PageView>,
}

fn busy() -> ApiError {
    ApiError::new(
        StatusCode::CONFLICT,
        "cannot change the project while a pipeline run is active",
    )
}

fn scene_error(error: koharu_scene::Error) -> ApiError {
    let status = match &error {
        koharu_scene::Error::Storage(storage) => match storage {
            koharu_storage::Error::Io(io) if io.kind() == io::ErrorKind::NotFound => {
                StatusCode::NOT_FOUND
            }
            koharu_storage::Error::Locked => StatusCode::CONFLICT,
            koharu_storage::Error::NotAProject
            | koharu_storage::Error::Codec(_)
            | koharu_storage::Error::UnsupportedFormat(_)
            | koharu_storage::Error::Invalid(_) => StatusCode::UNPROCESSABLE_ENTITY,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        },
        koharu_scene::Error::Codec(_) | koharu_scene::Error::Invalid(_) => {
            StatusCode::UNPROCESSABLE_ENTITY
        }
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    ApiError::new(status, error.to_string())
}

fn project_view(project: &OpenProject) -> Result<ProjectView, koharu_scene::Error> {
    Ok(ProjectView {
        name: project.name.clone(),
        path: project.path.display().to_string(),
        pages: pages_view(&project.session.snapshot())?,
    })
}

fn pages_view(snapshot: &Snapshot) -> Result<Vec<PageView>, koharu_scene::Error> {
    snapshot
        .pages()
        .enumerate()
        .map(|(index, page)| {
            let value = page.page()?;
            Ok(PageView {
                id: page.id().to_string(),
                index,
                width: value.width,
                height: value.height,
            })
        })
        .collect()
}

async fn get_project(State(state): State<AppState>) -> Result<Json<ProjectView>, ApiError> {
    let current = state.project.lock().await;
    let Some(project) = current.as_ref() else {
        return Err(ApiError::new(StatusCode::NOT_FOUND, "no project is open"));
    };
    Ok(Json(project_view(project).map_err(scene_error)?))
}

async fn open_project(
    State(state): State<AppState>,
    JsonBody(request): JsonBody<OpenProjectRequest>,
) -> Result<Json<ProjectView>, ApiError> {
    let create = request.create.unwrap_or(false);
    let path = PathBuf::from(request.path);
    if !path.is_absolute() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "project path must be absolute",
        ));
    }
    if create {
        if path.exists() && !path.is_dir() {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                format!("{} is not a directory", path.display()),
            ));
        }
    } else if !path.is_dir() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("project directory not found: {}", path.display()),
        ));
    }
    let mut current = state.project.lock().await;
    // Registration (`Operations::start`) happens under this same lock in
    // `start_pipeline`, so the busy check must follow the lock to be race-free.
    if state.operations.is_running() {
        return Err(busy());
    }
    // Dropping before the new open releases the storage lock the old session
    // holds, so re-opening the same directory cannot deadlock on itself.
    *current = None;
    let session = if create {
        Session::create(&path).await.map_err(|error| match &error {
            koharu_scene::Error::Storage(koharu_storage::Error::Invalid(_)) => ApiError::new(
                StatusCode::CONFLICT,
                format!("project already exists at {}", path.display()),
            ),
            _ => scene_error(error),
        })?
    } else {
        Session::open(&path).await.map_err(scene_error)?
    };
    let path = dunce::canonicalize(&path).unwrap_or(path);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .map_or_else(|| session.project_id().to_string(), ToOwned::to_owned);
    let project = OpenProject {
        session,
        path,
        name,
    };
    let view = project_view(&project).map_err(scene_error)?;
    *current = Some(project);
    Ok(Json(view))
}

struct PreparedPage {
    label: String,
    bytes: Arc<[u8]>,
    media_type: &'static str,
    width: u32,
    height: u32,
}

fn prepare_pages(paths: Vec<PathBuf>) -> Result<Vec<PreparedPage>, ApiError> {
    paths
        .iter()
        .map(|path| {
            let bytes = std::fs::read(path).map_err(|error| {
                let status = if error.kind() == io::ErrorKind::NotFound {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::UNPROCESSABLE_ENTITY
                };
                ApiError::new(
                    status,
                    format!("failed to read {}: {error}", path.display()),
                )
            })?;
            let reader = ImageReader::new(io::Cursor::new(&bytes))
                .with_guessed_format()
                .map_err(|error| {
                    ApiError::new(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!("cannot detect image format of {}: {error}", path.display()),
                    )
                })?;
            let media_type = match reader.format() {
                Some(ImageFormat::Png) => "image/png",
                Some(ImageFormat::Jpeg) => "image/jpeg",
                Some(ImageFormat::WebP) => "image/webp",
                _ => {
                    return Err(ApiError::new(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!("unsupported image format: {}", path.display()),
                    ));
                }
            };
            let (width, height) = reader.into_dimensions().map_err(|error| {
                ApiError::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!(
                        "cannot read image dimensions of {}: {error}",
                        path.display()
                    ),
                )
            })?;
            Ok(PreparedPage {
                label: path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("page")
                    .to_owned(),
                bytes: bytes.into(),
                media_type,
                width,
                height,
            })
        })
        .collect()
}

async fn add_pages(
    State(state): State<AppState>,
    JsonBody(request): JsonBody<AddPagesRequest>,
) -> Result<Json<AddPagesResponse>, ApiError> {
    if request.paths.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "paths must not be empty",
        ));
    }
    let paths = request.paths.iter().map(PathBuf::from).collect::<Vec<_>>();
    if let Some(path) = paths.iter().find(|path| !path.is_absolute()) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("page path must be absolute: {}", path.display()),
        ));
    }
    if state.operations.is_running() {
        return Err(busy());
    }
    let prepared = tokio::task::spawn_blocking(move || prepare_pages(paths))
        .await
        .map_err(|error| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))??;

    let mut current = state.project.lock().await;
    // Re-checked under the lock: a run can register while the page files were
    // being read, because `Operations::start` runs under this same lock.
    if state.operations.is_running() {
        return Err(busy());
    }
    let Some(project) = current.as_mut() else {
        return Err(ApiError::new(StatusCode::NOT_FOUND, "no project is open"));
    };
    let snapshot = project.session.snapshot();
    let order = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
    let source = AssetRole::new("source").map_err(scene_error)?;
    let index = request.index;
    let mut added = Vec::new();
    let patch = snapshot
        .patch(|edit| {
            let mut previous = None;
            for page in &prepared {
                let at = if let Some(previous) = previous {
                    At::After(previous)
                } else if let Some(index) = index {
                    order.get(index).copied().map_or(At::End, At::Before)
                } else {
                    At::End
                };
                let id = edit.add_page(
                    PageDraft::new(
                        page.label.clone(),
                        f64::from(page.width),
                        f64::from(page.height),
                    ),
                    at,
                )?;
                edit.set_asset(
                    id,
                    &source,
                    AssetInput::new(
                        Arc::clone(&page.bytes),
                        page.media_type,
                        AssetMetadata {
                            width: Some(page.width),
                            height: Some(page.height),
                            attributes: BTreeMap::new(),
                        },
                    ),
                )?;
                added.push(id);
                previous = Some(id);
            }
            Ok(())
        })
        .map_err(scene_error)?;
    project.session.commit(patch).await.map_err(scene_error)?;

    let snapshot = project.session.snapshot();
    let pages = snapshot
        .pages()
        .enumerate()
        .filter(|(_, page)| added.contains(&page.id()))
        .map(|(index, page)| {
            let value = page.page()?;
            Ok(PageView {
                id: page.id().to_string(),
                index,
                width: value.width,
                height: value.height,
            })
        })
        .collect::<Result<Vec<_>, koharu_scene::Error>>()
        .map_err(scene_error)?;
    Ok(Json(AddPagesResponse { pages }))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct StartPipelineRequest {
    stages: Vec<Stage>,
    pages: Option<Vec<EntityId>>,
    target_language: Option<Language>,
    system_prompt: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StartPipelineResponse {
    operation_id: String,
}

async fn start_pipeline(
    State(state): State<AppState>,
    JsonBody(request): JsonBody<StartPipelineRequest>,
) -> Result<(StatusCode, Json<StartPipelineResponse>), ApiError> {
    if request.pages.as_ref().is_some_and(Vec::is_empty) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "pages must not be empty",
        ));
    }
    let pipeline = state
        .pipeline_for(request.target_language, request.system_prompt.as_deref())
        .map_err(|error| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    let current = state.project.lock().await;
    let Some(project) = current.as_ref() else {
        return Err(ApiError::new(StatusCode::NOT_FOUND, "no project is open"));
    };
    let snapshot = project.session.snapshot();
    let stop = StopToken::default();
    let progress = Arc::new(StdMutex::new(VecDeque::new()));
    let sink: ProgressSink = {
        let progress = Arc::clone(&progress);
        Arc::new(move |event| push_progress(&progress, event))
    };
    let pipeline_request = PipelineRequest {
        operation: if request.stages.is_empty() {
            Operation::Full
        } else {
            Operation::Stages {
                stages: request.stages,
            }
        },
        scope: request.pages.map_or(Scope::Project, Scope::Pages),
        stop: stop.clone(),
        progress: Some(sink),
        ..PipelineRequest::default()
    };
    let id = uuid::Uuid::now_v7().to_string();
    if !state.operations.start(Job {
        id: id.clone(),
        status: JobStatus::Running,
        message: None,
        completed: 0,
        total: 0,
        elapsed: None,
        started: Instant::now(),
        progress,
        stop,
    }) {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "a pipeline run is already active",
        ));
    }
    let runner = state.runner.clone();
    let committer = ProjectCommitter {
        project: Arc::clone(&state.project),
    };
    let job_id = id.clone();
    let state = Arc::clone(&state);
    tokio::spawn(async move {
        let outcome = (runner.0)(pipeline, snapshot, pipeline_request, committer).await;
        state.operations.finish(&job_id, outcome);
    });
    Ok((
        StatusCode::ACCEPTED,
        Json(StartPipelineResponse { operation_id: id }),
    ))
}

async fn list_operations(State(state): State<AppState>) -> Json<Vec<OperationView>> {
    Json(state.operations.view())
}

async fn stop_operation(State(state): State<AppState>, Path(id): Path<String>) -> StatusCode {
    state.operations.stop(&id);
    StatusCode::NO_CONTENT
}

async fn page_image(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let current = state.project.lock().await;
    let Some(project) = current.as_ref() else {
        return Err(ApiError::new(StatusCode::NOT_FOUND, "no project is open"));
    };
    let snapshot = project.session.snapshot();
    drop(current);
    let Some(page) = snapshot.pages().find(|page| page.id().to_string() == id) else {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("page not found: {id}"),
        ));
    };
    let source = AssetRole::new("source").map_err(scene_error)?;
    let Some(asset) = snapshot.asset(page.id(), &source).map_err(scene_error)? else {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("page {id} has no source image"),
        ));
    };
    let bytes = snapshot.read_blob(asset.blob).await.map_err(scene_error)?;
    Ok(([(header::CONTENT_TYPE, asset.media_type)], bytes).into_response())
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExportFormat {
    Png,
    Psd,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportRequest {
    pages: Option<Vec<EntityId>>,
    dir: Option<String>,
    format: Option<ExportFormat>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExportResponse {
    files: Vec<String>,
}

/// Renderer and Rasterizer for exports, built once per process on first use:
/// `Rasterizer::new` acquires a GPU device and `Renderer::new` spawns a
/// typesetting-config watcher, so neither is paid at startup or by an export
/// that never happens.
struct ExportEngine {
    renderer: Renderer,
    rasterizer: Arc<Rasterizer>,
}

/// One parked watcher per export-engine construction, at most one per process.
async fn export_engine(state: &AppState) -> Result<&ExportEngine, ApiError> {
    state
        .export
        .get_or_try_init(|| async {
            Ok::<_, ApiError>(ExportEngine {
                renderer: Renderer::new().map_err(engine_error)?,
                rasterizer: Arc::new(Rasterizer::new().map_err(engine_error)?),
            })
        })
        .await
}

fn engine_error(error: impl std::error::Error + Send + Sync + 'static) -> ApiError {
    // `{:#}` walks the anyhow source chain; Display alone hides the root cause.
    let error = anyhow::Error::new(error);
    ApiError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("export failed: {error:#}"),
    )
}

/// File stem for an exported page: 1-based position plus the sanitized page
/// label, matching the desktop export naming.
fn export_stem(index: usize, label: &str) -> String {
    let name = label
        .trim()
        .trim_end_matches(|character: char| character == '.' || character.is_whitespace());
    let name = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
    let name = name
        .chars()
        .map(|character| {
            if matches!(
                character,
                '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'
            ) {
                '_'
            } else {
                character
            }
        })
        .collect::<String>();
    format!(
        "{:04}_{}",
        index + 1,
        if name.is_empty() { "page" } else { &name }
    )
}

/// Export renders the selected pages from an immutable snapshot and writes
/// them outside project storage, so it is allowed while a pipeline run is
/// active; the result may be a mid-run partial render.
async fn export(
    State(state): State<AppState>,
    JsonBody(request): JsonBody<ExportRequest>,
) -> Result<Json<ExportResponse>, ApiError> {
    if request.pages.as_ref().is_some_and(Vec::is_empty) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "pages must not be empty",
        ));
    }
    let format = request.format.unwrap_or(ExportFormat::Png);
    let current = state.project.lock().await;
    let Some(project) = current.as_ref() else {
        return Err(ApiError::new(StatusCode::NOT_FOUND, "no project is open"));
    };
    let directory = request
        .dir
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| project.path.join("exports"));
    if !directory.is_absolute() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("export path must be absolute: {}", directory.display()),
        ));
    }
    let snapshot = project.session.snapshot();
    drop(current);
    let pages = match request.pages {
        Some(pages) => pages,
        None => snapshot.pages().map(|page| page.id()).collect(),
    };
    if let Some(missing) = pages
        .iter()
        .find(|id| snapshot.page(**id).map_err(scene_error).is_err())
    {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("page not found: {missing}"),
        ));
    }
    if pages.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "there are no pages to export",
        ));
    }
    tokio::fs::create_dir_all(&directory)
        .await
        .map_err(|error| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                format!("cannot create {}: {error}", directory.display()),
            )
        })?;
    let engine = export_engine(&state).await?;
    let mut files = Vec::with_capacity(pages.len());
    for (index, page) in pages.iter().enumerate() {
        let label = snapshot
            .page(*page)
            .and_then(|page| page.page())
            .map_err(scene_error)?
            .label;
        let path = directory.join(match format {
            ExportFormat::Png => format!("{}.png", export_stem(index, &label)),
            ExportFormat::Psd => format!("{}.psd", export_stem(index, &label)),
        });
        let frame = engine
            .renderer
            .render(&snapshot, *page)
            .await
            .map_err(engine_error)?;
        let bytes = match format {
            ExportFormat::Png => {
                let raster_frame = frame.raster_frame().map_err(engine_error)?;
                let rasterizer = Arc::clone(&engine.rasterizer);
                tokio::task::spawn_blocking(move || {
                    rasterizer
                        .rasterize(&raster_frame, RasterOptions::default())
                        .map(|raster| raster.image)
                        .map_err(engine_error)
                        .and_then(encode_png)
                })
                .await
                .map_err(|error| {
                    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
                })??
            }
            ExportFormat::Psd => export_page(
                Arc::clone(&engine.rasterizer),
                &snapshot,
                &frame,
                &PsdExportOptions::default(),
            )
            .await
            .map_err(engine_error)?,
        };
        write_atomic(path.clone(), bytes).await?;
        files.push(path.display().to_string());
    }
    Ok(Json(ExportResponse { files }))
}

/// PNG encode at the desktop's quality settings.
fn encode_png(image: image::RgbaImage) -> Result<Vec<u8>, ApiError> {
    let mut bytes = Vec::new();
    PngEncoder::new_with_quality(
        io::Cursor::new(&mut bytes),
        CompressionType::Best,
        FilterType::Adaptive,
    )
    .write_image(
        image.as_raw(),
        image.width(),
        image.height(),
        ExtendedColorType::Rgba8,
    )
    .map_err(engine_error)?;
    Ok(bytes)
}

/// Writes through a unique temp file and renames, so concurrent exports of
/// one page cannot interleave into the same destination path.
async fn write_atomic(path: PathBuf, bytes: Vec<u8>) -> Result<(), ApiError> {
    tokio::task::spawn_blocking(move || {
        let io_error = |error: io::Error| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                format!("failed to write {}: {error}", path.display()),
            )
        };
        let temporary = path.with_extension(format!("{}.tmp", uuid::Uuid::now_v7()));
        std::fs::write(&temporary, bytes).map_err(io_error)?;
        std::fs::rename(&temporary, &path).map_err(|error| {
            let _ = std::fs::remove_file(&temporary);
            io_error(error)
        })?;
        Ok::<_, ApiError>(())
    })
    .await
    .map_err(|error| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))??;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        At, Committer, Config, EntityId, Language, Operation, PageDraft, Pipeline, PipelineConfig,
        PipelineRequest, Progress, ProjectCommitter, ProvidersConfig, Report, RunStatus, Runner,
        Scope, ServerState, Snapshot, Stage, StageOutput, export_stem, memo_hit, router,
    };
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use koharu_ml::Device;
    use serde_json::{Value, json};
    use std::{
        fs,
        future::Future,
        io::Cursor,
        path::{Path, PathBuf},
        sync::{Arc, Mutex as StdMutex},
        time::Duration,
    };
    use tower::ServiceExt;

    type App = axum::Router;

    fn test_pipeline() -> Pipeline {
        Pipeline::from_config(
            Config::memory(PipelineConfig::default()),
            Config::memory(ProvidersConfig::default()),
            Device::cpu(),
        )
        .expect("a memory-config pipeline needs no weights")
    }

    fn app_with(runner: Runner) -> App {
        router(Arc::new(ServerState::new(
            test_pipeline(),
            Device::cpu(),
            runner,
        )))
    }

    fn app() -> App {
        app_with(Runner::execute())
    }

    async fn send(app: &App, request: Request<Body>) -> (axum::http::StatusCode, Value) {
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json = if body.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&body).unwrap()
        };
        (status, json)
    }

    fn post_json(uri: &str, body: Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn get(uri: &str) -> Request<Body> {
        Request::builder().uri(uri).body(Body::empty()).unwrap()
    }

    fn delete(uri: &str) -> Request<Body> {
        Request::builder()
            .method("DELETE")
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    }

    fn write_png(dir: &Path, name: &str, width: u32, height: u32) -> PathBuf {
        let path = dir.join(name);
        image::RgbaImage::from_pixel(width, height, image::Rgba([10, 20, 30, 255]))
            .save_with_format(&path, image::ImageFormat::Png)
            .unwrap();
        path
    }

    fn runner_with<F, Fut>(f: F) -> Runner
    where
        F: Fn(Pipeline, Snapshot, PipelineRequest, ProjectCommitter) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = anyhow::Result<Report>> + Send + 'static,
    {
        Runner(Arc::new(move |pipeline, snapshot, request, committer| {
            Box::pin(f(pipeline, snapshot, request, committer))
        }))
    }

    fn completing_runner() -> Runner {
        runner_with(|_pipeline, snapshot, request, mut committer| async move {
            let pages = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
            let sink = request.progress.expect("run carries a progress sink");
            sink(Progress::Started {
                pages: pages.clone(),
                stages: vec![Stage::Ocr],
            });
            let patch = snapshot.patch(|edit| {
                edit.add_page(PageDraft::new("translated", 1.0, 1.0), At::End)?;
                Ok(())
            })?;
            let committed = committer
                .commit(StageOutput {
                    page: pages[0],
                    stage: Stage::Ocr,
                    patch,
                })
                .await?;
            sink(Progress::Finished {
                page: pages[0],
                stage: Stage::Ocr,
                model: "fake-ocr".to_owned(),
                elapsed: Duration::from_millis(5),
            });
            Ok(Report {
                status: RunStatus::Completed,
                base: snapshot.revision(),
                final_revision: committed.revision(),
                completed: 1,
                total: 1,
                elapsed: Duration::from_millis(5),
            })
        })
    }

    fn recording_runner(recorded: Arc<StdMutex<Vec<PipelineRequest>>>) -> Runner {
        runner_with(move |_pipeline, snapshot, request, _committer| {
            let recorded = Arc::clone(&recorded);
            async move {
                recorded
                    .lock()
                    .expect("recorded requests are poisoned")
                    .push(request);
                Ok(Report {
                    status: RunStatus::Completed,
                    base: snapshot.revision(),
                    final_revision: snapshot.revision(),
                    completed: 0,
                    total: 0,
                    elapsed: Duration::from_millis(1),
                })
            }
        })
    }

    fn stopping_runner() -> Runner {
        runner_with(|_pipeline, snapshot, request, _committer| async move {
            let pages = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
            if let Some(sink) = &request.progress {
                sink(Progress::Started {
                    pages,
                    stages: vec![Stage::Ocr],
                });
            }
            while !request.stop.stopped() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            Ok(Report {
                status: RunStatus::Stopped,
                base: snapshot.revision(),
                final_revision: snapshot.revision(),
                completed: 0,
                total: 2,
                elapsed: Duration::from_millis(10),
            })
        })
    }

    fn failing_runner() -> Runner {
        runner_with(|_pipeline, _snapshot, _request, _committer| async {
            Err(anyhow::anyhow!("model weights are missing"))
        })
    }

    async fn open_project_with_pages(app: &App, root: &Path) -> Vec<String> {
        let first = write_png(root, "page-a.png", 3, 4);
        let second = write_png(root, "page-b.png", 8, 2);
        let (status, _) = send(
            app,
            post_json(
                "/api/v1/project/open",
                json!({"path": root.join("project").display().to_string(), "create": true}),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let (status, body) = send(
            app,
            post_json(
                "/api/v1/project/pages",
                json!({"paths": [first.display().to_string(), second.display().to_string()]}),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        body["pages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|page| page["id"].as_str().unwrap().to_owned())
            .collect()
    }

    async fn wait_status(app: &App, id: &str, expected: &str) -> Value {
        for _ in 0..400 {
            let (_, operations) = send(app, get("/api/v1/operations")).await;
            let entry = operations
                .as_array()
                .and_then(|entries| entries.iter().find(|entry| entry["id"] == id))
                .cloned();
            if let Some(entry) = entry
                && entry["status"] == expected
            {
                return entry;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("operation {id} never reached status {expected}");
    }

    #[tokio::test]
    async fn health_reports_ok_status_and_package_version() {
        let app = app();

        let (status, body) = send(&app, get("/api/v1/health")).await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["status"], "ok");
        assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn unknown_route_returns_json_error_envelope() {
        let app = app();

        let (status, body) = send(&app, get("/nope")).await;
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
        assert_eq!(body["status"], 404);
        assert_eq!(body["message"], "no route for /nope");
    }

    #[tokio::test]
    async fn created_project_serves_added_pages_with_indexes_and_dimensions() {
        let root = tempfile::tempdir().unwrap();
        let first = write_png(root.path(), "page-a.png", 3, 4);
        let second = write_png(root.path(), "page-b.png", 8, 2);
        let app = app();

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/project/open",
                json!({ "path": root.path().join("project"), "create": true }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["name"], "project");
        assert_eq!(body["pages"], json!([]));

        let (status, body) = send(
            &app,
            post_json("/api/v1/project/pages", json!({ "paths": [first, second] })),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["pages"][0]["index"], 0);
        assert_eq!(body["pages"][0]["width"], 3.0);
        assert_eq!(body["pages"][0]["height"], 4.0);
        assert_eq!(body["pages"][1]["index"], 1);
        assert_eq!(body["pages"][1]["width"], 8.0);
        assert_eq!(body["pages"][1]["height"], 2.0);
        assert!(
            body["pages"][0]["id"]
                .as_str()
                .is_some_and(|id| !id.is_empty())
        );

        let (status, body) = send(&app, get("/api/v1/project")).await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["name"], "project");
        assert_eq!(body["pages"].as_array().unwrap().len(), 2);
        assert_eq!(body["pages"][0]["width"], 3.0);
        assert_eq!(body["pages"][1]["height"], 2.0);
    }

    #[tokio::test]
    async fn reopening_from_disk_preserves_committed_pages() {
        let root = tempfile::tempdir().unwrap();
        let page = write_png(root.path(), "page.png", 5, 6);
        let app = app();
        let project_path = root.path().join("project");

        send(
            &app,
            post_json(
                "/api/v1/project/open",
                json!({ "path": project_path, "create": true }),
            ),
        )
        .await;
        send(
            &app,
            post_json("/api/v1/project/pages", json!({ "paths": [page] })),
        )
        .await;

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/project/open",
                json!({ "path": project_path, "create": false }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["pages"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn creating_over_existing_project_conflicts() {
        let root = tempfile::tempdir().unwrap();
        let app = app();
        let project_path = root.path().join("project");

        let (status, _) = send(
            &app,
            post_json(
                "/api/v1/project/open",
                json!({ "path": project_path.clone(), "create": true }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/project/open",
                json!({ "path": project_path, "create": true }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::CONFLICT);
        assert_eq!(body["status"], 409);
        assert!(!body["message"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn opening_missing_directory_returns_404() {
        let root = tempfile::tempdir().unwrap();
        let app = app();

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/project/open",
                json!({ "path": root.path().join("missing") }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
        assert_eq!(body["status"], 404);
        assert!(!body["message"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn opening_non_project_directory_returns_422() {
        let root = tempfile::tempdir().unwrap();
        let empty = root.path().join("empty");
        std::fs::create_dir(&empty).unwrap();
        let app = app();

        let (status, body) = send(
            &app,
            post_json("/api/v1/project/open", json!({ "path": empty })),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["status"], 422);
        assert!(!body["message"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn adding_pages_without_open_project_returns_404() {
        let root = tempfile::tempdir().unwrap();
        let page = write_png(root.path(), "page.png", 2, 2);
        let app = app();

        let (status, body) = send(
            &app,
            post_json("/api/v1/project/pages", json!({ "paths": [page] })),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
        assert_eq!(body["status"], 404);
    }

    #[tokio::test]
    async fn relative_page_path_returns_400() {
        let app = app();

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/project/pages",
                json!({ "paths": ["relative/page.png"] }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["status"], 400);
    }

    #[tokio::test]
    async fn missing_page_file_returns_404() {
        let root = tempfile::tempdir().unwrap();
        let app = app();
        send(
            &app,
            post_json(
                "/api/v1/project/open",
                json!({ "path": root.path().join("project"), "create": true }),
            ),
        )
        .await;

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/project/pages",
                json!({ "paths": [root.path().join("absent.png")] }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
        assert_eq!(body["status"], 404);
    }

    #[tokio::test]
    async fn malformed_json_body_returns_error_envelope() {
        let app = app();

        let request = Request::builder()
            .method("POST")
            .uri("/api/v1/project/open")
            .header("content-type", "application/json")
            .body(Body::from("{not json"))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok());
        assert_eq!(content_type, Some("application/json"));
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["status"], 400);
        assert!(!value["message"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn inserting_pages_at_index_reorders_committed_pages() {
        let root = tempfile::tempdir().unwrap();
        let first = write_png(root.path(), "first.png", 2, 2);
        let second = write_png(root.path(), "second.png", 3, 3);
        let inserted_a = write_png(root.path(), "inserted-a.png", 4, 4);
        let inserted_b = write_png(root.path(), "inserted-b.png", 5, 5);
        let app = app();

        send(
            &app,
            post_json(
                "/api/v1/project/open",
                json!({ "path": root.path().join("project"), "create": true }),
            ),
        )
        .await;
        send(
            &app,
            post_json("/api/v1/project/pages", json!({ "paths": [first, second] })),
        )
        .await;

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/project/pages",
                json!({ "paths": [inserted_a, inserted_b], "index": 1 }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["pages"][0]["index"], 1);
        assert_eq!(body["pages"][0]["width"], 4.0);
        assert_eq!(body["pages"][1]["index"], 2);
        assert_eq!(body["pages"][1]["width"], 5.0);

        let (_, project) = send(&app, get("/api/v1/project")).await;
        assert_eq!(project["pages"][0]["width"], 2.0);
        assert_eq!(project["pages"][1]["width"], 4.0);
        assert_eq!(project["pages"][2]["width"], 5.0);
        assert_eq!(project["pages"][3]["width"], 3.0);
    }

    #[tokio::test]
    async fn pipeline_run_completes_and_committer_persists_patches() {
        let root = tempfile::tempdir().unwrap();
        let app = app_with(completing_runner());
        let pages = open_project_with_pages(&app, root.path()).await;

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/pipelines",
                json!({ "stages": ["ocr"], "pages": [pages[0]] }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::ACCEPTED);
        let id = body["operationId"].as_str().unwrap().to_owned();

        let entry = wait_status(&app, &id, "completed").await;
        assert_eq!(entry["completed"], 1);
        assert_eq!(entry["total"], 1);
        assert!(entry["elapsed"].as_f64().unwrap() > 0.0);
        assert_eq!(entry["recentProgress"][0]["event"], "started");
        assert_eq!(entry["recentProgress"][0]["pages"][0], pages[0]);
        assert_eq!(entry["recentProgress"][1]["event"], "finished");
        assert_eq!(entry["recentProgress"][1]["stage"], "ocr");
        assert_eq!(entry["recentProgress"][1]["model"], "fake-ocr");

        let (_, project) = send(&app, get("/api/v1/project")).await;
        assert_eq!(project["pages"].as_array().unwrap().len(), 3);
        assert_eq!(project["pages"][2]["width"], 1.0);
        assert_eq!(project["pages"][2]["height"], 1.0);
    }

    #[tokio::test]
    async fn deleting_a_running_operation_stops_it_and_is_idempotent() {
        let root = tempfile::tempdir().unwrap();
        let app = app_with(stopping_runner());
        open_project_with_pages(&app, root.path()).await;

        let (status, body) = send(
            &app,
            post_json("/api/v1/pipelines", json!({ "stages": [] })),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::ACCEPTED);
        let id = body["operationId"].as_str().unwrap().to_owned();

        let (status, _) = send(&app, delete(&format!("/api/v1/operations/{id}"))).await;
        assert_eq!(status, axum::http::StatusCode::NO_CONTENT);
        let (status, _) = send(&app, delete(&format!("/api/v1/operations/{id}"))).await;
        assert_eq!(status, axum::http::StatusCode::NO_CONTENT);
        let (status, _) = send(&app, delete("/api/v1/operations/never-existed")).await;
        assert_eq!(status, axum::http::StatusCode::NO_CONTENT);

        let entry = wait_status(&app, &id, "stopped").await;
        assert_eq!(entry["completed"], 0);
        assert_eq!(entry["total"], 2);
    }

    #[tokio::test]
    async fn failed_pipeline_run_reports_failure_message() {
        let root = tempfile::tempdir().unwrap();
        let app = app_with(failing_runner());
        open_project_with_pages(&app, root.path()).await;

        let (status, body) = send(
            &app,
            post_json("/api/v1/pipelines", json!({ "stages": ["ocr"] })),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::ACCEPTED);
        let id = body["operationId"].as_str().unwrap().to_owned();

        let entry = wait_status(&app, &id, "failed").await;
        assert_eq!(entry["message"], "model weights are missing");
        assert_eq!(entry["completed"], 0);
        assert_eq!(entry["total"], 0);
        let frozen = entry["elapsed"].as_f64().unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        let (_, operations) = send(&app, get("/api/v1/operations")).await;
        assert_eq!(operations[0]["elapsed"].as_f64().unwrap(), frozen);
    }

    #[tokio::test]
    async fn second_run_and_project_changes_conflict_while_pipeline_is_active() {
        let root = tempfile::tempdir().unwrap();
        let app = app_with(stopping_runner());
        open_project_with_pages(&app, root.path()).await;

        let (status, body) = send(
            &app,
            post_json("/api/v1/pipelines", json!({ "stages": [] })),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::ACCEPTED);
        let id = body["operationId"].as_str().unwrap().to_owned();

        let (status, body) = send(
            &app,
            post_json("/api/v1/pipelines", json!({ "stages": [] })),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::CONFLICT);
        assert_eq!(body["status"], 409);
        let (status, _) = send(
            &app,
            post_json(
                "/api/v1/project/open",
                json!({ "path": root.path().join("other").display().to_string(), "create": true }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::CONFLICT);
        let (status, _) = send(
            &app,
            post_json(
                "/api/v1/project/pages",
                json!({ "paths": [root.path().join("x.png").display().to_string()] }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::CONFLICT);

        send(&app, delete(&format!("/api/v1/operations/{id}"))).await;
        wait_status(&app, &id, "stopped").await;
    }

    #[tokio::test]
    async fn pipeline_request_maps_stages_and_pages() {
        let root = tempfile::tempdir().unwrap();
        let recorded = Arc::new(StdMutex::new(Vec::new()));
        let app = app_with(recording_runner(Arc::clone(&recorded)));
        let pages = open_project_with_pages(&app, root.path()).await;

        for body in [
            json!({ "stages": [] }),
            json!({ "stages": ["ocr"] }),
            json!({ "stages": ["ocr"], "pages": [pages[0]] }),
        ] {
            let (status, body) = send(&app, post_json("/api/v1/pipelines", body)).await;
            assert_eq!(status, axum::http::StatusCode::ACCEPTED);
            wait_status(&app, body["operationId"].as_str().unwrap(), "completed").await;
        }

        let requests = recorded.lock().expect("recorded requests are poisoned");
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[0].operation, Operation::Full);
        assert_eq!(requests[0].scope, Scope::Project);
        assert_eq!(
            requests[1].operation,
            Operation::Stages {
                stages: vec![Stage::Ocr]
            }
        );
        let page: EntityId = serde_json::from_value(json!(pages[0])).unwrap();
        assert_eq!(requests[2].scope, Scope::Pages(vec![page]));
    }

    #[tokio::test]
    async fn memo_reuses_the_pipeline_for_matching_overrides_and_rebuilds_on_new_ones() {
        let memo = Some(((Some(Language::English), None), test_pipeline()));

        assert!(memo_hit(&memo, &(Some(Language::English), None)).is_some());
        assert!(
            memo_hit(
                &memo,
                &(Some(Language::English), Some("keep names".to_owned()))
            )
            .is_none()
        );
        assert!(memo_hit(&None, &(Some(Language::English), None)).is_none());
    }

    #[tokio::test]
    async fn unknown_stage_string_returns_422_envelope() {
        let app = app();

        let (status, body) = send(
            &app,
            post_json("/api/v1/pipelines", json!({ "stages": ["teleport"] })),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["status"], 422);
        assert!(!body["message"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn empty_pages_list_returns_400_without_starting_a_job() {
        let root = tempfile::tempdir().unwrap();
        let app = app();
        open_project_with_pages(&app, root.path()).await;

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/pipelines",
                json!({ "stages": ["ocr"], "pages": [] }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["status"], 400);

        let (_, operations) = send(&app, get("/api/v1/operations")).await;
        assert_eq!(operations.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn page_image_serves_the_stored_source_blob() {
        let root = tempfile::tempdir().unwrap();
        let app = app();
        let first = write_png(root.path(), "page-a.png", 3, 4);
        let (status, _) = send(
            &app,
            post_json(
                "/api/v1/project/open",
                json!({ "path": root.path().join("project").display().to_string(), "create": true }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/project/pages",
                json!({ "paths": [first.display().to_string()] }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let id = body["pages"][0]["id"].as_str().unwrap().to_owned();

        let response = app
            .clone()
            .oneshot(get(&format!("/api/v1/pages/{id}/image")))
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "image/png");
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(&bytes[..], std::fs::read(&first).unwrap().as_slice());

        let (status, body) = send(&app, get("/api/v1/pages/zzz/image")).await;
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
        assert_eq!(body["status"], 404);
    }

    #[tokio::test]
    async fn starting_pipeline_without_project_returns_404() {
        let app = app();

        let (status, body) = send(
            &app,
            post_json("/api/v1/pipelines", json!({ "stages": ["ocr"] })),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
        assert_eq!(body["status"], 404);

        let (_, operations) = send(&app, get("/api/v1/operations")).await;
        assert_eq!(operations.as_array().unwrap().len(), 0);
    }

    fn png_dimensions(path: &Path) -> (u32, u32) {
        let bytes = fs::read(path).unwrap();
        assert!(!bytes.is_empty());
        image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .unwrap()
            .into_dimensions()
            .unwrap()
    }

    #[tokio::test]
    async fn export_renders_numbered_files_matching_the_page_dimensions() {
        let root = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let app = app();
        open_project_with_pages(&app, root.path()).await;

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/export",
                json!({"dir": out.path().display().to_string()}),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let files = body["files"].as_array().unwrap();
        assert_eq!(files.len(), 2);
        let names: Vec<String> = files
            .iter()
            .map(|file| {
                Path::new(file.as_str().unwrap())
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(names, ["0001_page-a.png", "0002_page-b.png"]);
        assert_eq!(
            png_dimensions(Path::new(files[0].as_str().unwrap())),
            (3, 4)
        );
        assert_eq!(
            png_dimensions(Path::new(files[1].as_str().unwrap())),
            (8, 2)
        );
    }

    #[tokio::test]
    async fn export_honors_the_page_selection() {
        let root = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let app = app();
        let pages = open_project_with_pages(&app, root.path()).await;

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/export",
                json!({"dir": out.path().display().to_string(), "pages": [pages[1]]}),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let files = body["files"].as_array().unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].as_str().unwrap().ends_with("0001_page-b.png"));
        assert_eq!(
            png_dimensions(Path::new(files[0].as_str().unwrap())),
            (8, 2)
        );
    }

    #[tokio::test]
    async fn export_defaults_to_the_project_exports_directory() {
        let root = tempfile::tempdir().unwrap();
        let app = app();
        open_project_with_pages(&app, root.path()).await;

        let (status, body) = send(&app, post_json("/api/v1/export", json!({}))).await;
        assert_eq!(status, axum::http::StatusCode::OK);
        for file in body["files"].as_array().unwrap() {
            let file = Path::new(file.as_str().unwrap());
            assert_eq!(
                file.parent().unwrap(),
                &root.path().join("project").join("exports")
            );
            assert!(file.is_file());
        }
    }

    #[tokio::test]
    async fn export_writes_psd_files_when_requested() {
        let root = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let app = app();
        open_project_with_pages(&app, root.path()).await;

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/export",
                json!({"dir": out.path().display().to_string(), "format": "psd"}),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let files = body["files"].as_array().unwrap();
        assert_eq!(files.len(), 2);
        for (file, (page_width, page_height)) in files.iter().zip([(3u32, 4u32), (8, 2)]) {
            let bytes = fs::read(file.as_str().unwrap()).unwrap();
            assert!(!bytes.is_empty());
            // PSD header: "8BPS", version 1, 6 reserved bytes, channels,
            // height, width, depth, mode (koharu-psd `write_header`).
            assert_eq!(&bytes[..4], b"8BPS");
            assert_eq!(&bytes[4..6], [0, 1]);
            assert!(bytes[6..12].iter().all(|byte| *byte == 0));
            let channels = u16::from_be_bytes([bytes[12], bytes[13]]);
            let height = u32::from_be_bytes(bytes[14..18].try_into().unwrap());
            let width = u32::from_be_bytes(bytes[18..22].try_into().unwrap());
            assert_eq!(&bytes[22..24], [0, 8]);
            assert_eq!(&bytes[24..26], [0, 3]);
            assert!((3..=4).contains(&channels));
            assert_eq!((width, height), (page_width, page_height));
        }
    }

    #[tokio::test]
    async fn export_without_project_returns_404() {
        let app = app();

        let (status, body) = send(&app, post_json("/api/v1/export", json!({}))).await;
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
        assert_eq!(body["message"], "no project is open");
    }

    #[tokio::test]
    async fn export_rejects_a_relative_dir() {
        let root = tempfile::tempdir().unwrap();
        let app = app();
        open_project_with_pages(&app, root.path()).await;

        let (status, body) = send(&app, post_json("/api/v1/export", json!({"dir": "out"}))).await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["message"], "export path must be absolute: out");
    }

    #[tokio::test]
    async fn export_rejects_an_unwritable_dir() {
        let root = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let app = app();
        open_project_with_pages(&app, root.path()).await;

        let blocker = write_png(out.path(), "blocker", 1, 1);
        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/export",
                json!({"dir": blocker.join("exports").display().to_string()}),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            body["message"]
                .as_str()
                .unwrap()
                .starts_with("cannot create ")
        );
    }

    #[tokio::test]
    async fn export_rejects_an_unknown_page_id() {
        let root = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let app = app();
        open_project_with_pages(&app, root.path()).await;

        let (status, body) = send(
            &app,
            post_json(
                "/api/v1/export",
                json!({
                    "dir": out.path().display().to_string(),
                    "pages": ["00000000-0000-0000-0000-000000000000"],
                }),
            ),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
        assert_eq!(
            body["message"],
            "page not found: 00000000-0000-0000-0000-000000000000"
        );
    }

    #[test]
    fn export_stem_numbers_pages_and_sanitizes_labels() {
        assert_eq!(export_stem(0, "page-a"), "0001_page-a");
        assert_eq!(export_stem(1, "  weird:name.png "), "0002_weird_name");
        assert_eq!(export_stem(2, "..."), "0003_page");
    }
}
