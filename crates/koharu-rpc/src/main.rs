use std::{
    collections::BTreeMap,
    io,
    net::{IpAddr, Ipv4Addr},
    path::PathBuf,
    sync::Arc,
};

use anyhow::Result;
use axum::{
    Json, Router,
    extract::{FromRequest, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use image::{ImageFormat, ImageReader};
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, PageDraft, Session, Snapshot};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::{net::TcpListener, sync::Mutex};

const DEFAULT_PORT: u16 = 4100;

#[derive(Debug, Parser)]
#[command(version, about = "Koharu headless HTTP and MCP server")]
struct Arguments {
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    host: IpAddr,

    #[arg(long, value_name = "PORT", default_value_t = DEFAULT_PORT)]
    port: u16,
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

#[derive(Default)]
struct ServerState {
    project: Mutex<Option<OpenProject>>,
}

type AppState = Arc<ServerState>;

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();

    if !arguments.host.is_loopback() {
        eprintln!(
            "warning: binding {} exposes the unauthenticated API beyond loopback",
            arguments.host
        );
    }

    let listener = bind(&arguments.host, arguments.port).await?;
    println!(
        "koharu-rpc listening on http://{}:{}",
        arguments.host, arguments.port
    );

    axum::serve(listener, router(Arc::new(ServerState::default())))
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
    let prepared = tokio::task::spawn_blocking(move || prepare_pages(paths))
        .await
        .map_err(|error| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))??;

    let mut current = state.project.lock().await;
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

#[cfg(test)]
mod tests {
    use super::router;
    use crate::ServerState;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use serde_json::{Value, json};
    use std::{path::Path, path::PathBuf, sync::Arc};
    use tower::ServiceExt;

    type App = axum::Router;

    async fn send(app: &App, request: Request<Body>) -> (axum::http::StatusCode, Value) {
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&body).unwrap())
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

    fn write_png(dir: &Path, name: &str, width: u32, height: u32) -> PathBuf {
        let path = dir.join(name);
        image::RgbaImage::from_pixel(width, height, image::Rgba([10, 20, 30, 255]))
            .save_with_format(&path, image::ImageFormat::Png)
            .unwrap();
        path
    }

    #[tokio::test]
    async fn health_reports_ok_status_and_package_version() {
        let app = router(Arc::new(ServerState::default()));

        let (status, body) = send(&app, get("/api/v1/health")).await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["status"], "ok");
        assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn unknown_route_returns_json_error_envelope() {
        let app = router(Arc::new(ServerState::default()));

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
        let app = router(Arc::new(ServerState::default()));

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
        let app = router(Arc::new(ServerState::default()));
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
        let app = router(Arc::new(ServerState::default()));
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
        let app = router(Arc::new(ServerState::default()));

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
        let app = router(Arc::new(ServerState::default()));

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
        let app = router(Arc::new(ServerState::default()));

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
        let app = router(Arc::new(ServerState::default()));

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
        let app = router(Arc::new(ServerState::default()));
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
        let app = router(Arc::new(ServerState::default()));

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
        let app = router(Arc::new(ServerState::default()));

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
}
