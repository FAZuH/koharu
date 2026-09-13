//! MCP transport: the six service-core operations exposed as MCP tools over
//! the streamable HTTP transport at `/mcp`, sharing the same functions the
//! HTTP handlers call.

use std::{net::IpAddr, sync::Arc};

use axum::Router;
use rmcp::{
    ErrorData, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};

use crate::{
    AddPagesRequest, ApiError, AppState, ExportRequest, OpenProjectRequest, StartPipelineRequest,
    add_pages_service, export_service, list_operations_service, open_project_service,
    start_pipeline_service, stop_operation_service,
};

pub(crate) struct KoharuTools {
    state: AppState,
    tool_router: ToolRouter<Self>,
}

impl KoharuTools {
    pub(crate) fn new(state: AppState) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    fn result(value: impl serde::Serialize) -> Result<CallToolResult, ErrorData> {
        let value = rmcp::serde_json::to_value(value).map_err(|error| {
            ErrorData::internal_error(format!("result serialization failed: {error}"), None)
        })?;
        Ok(CallToolResult::structured(value))
    }

    /// Service-core failures are tool-level errors: the request reached the
    /// tool, so the caller sees the message as `isError` content instead of a
    /// JSON-RPC protocol error.
    fn failure(error: ApiError) -> CallToolResult {
        CallToolResult::error(vec![ContentBlock::text(format!(
            "{}: {}",
            error.status, error.message
        ))])
    }
}

#[tool_router]
impl KoharuTools {
    #[tool(
        name = "koharu.open_project",
        description = "Open (or create) a Koharu project from an absolute directory path"
    )]
    pub(crate) async fn open_project(
        &self,
        Parameters(request): Parameters<OpenProjectRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        match open_project_service(&self.state, request).await {
            Ok(view) => Self::result(view),
            Err(error) => Ok(Self::failure(error)),
        }
    }

    #[tool(
        name = "koharu.add_pages",
        description = "Add image files (PNG, JPEG, WebP) to the open project as pages"
    )]
    pub(crate) async fn add_pages(
        &self,
        Parameters(request): Parameters<AddPagesRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        match add_pages_service(&self.state, request).await {
            Ok(response) => Self::result(response),
            Err(error) => Ok(Self::failure(error)),
        }
    }

    #[tool(
        name = "koharu.start_pipeline",
        description = "Start a pipeline run (OCR, translation, inpainting) on the open project and return its operation id"
    )]
    pub(crate) async fn start_pipeline(
        &self,
        Parameters(request): Parameters<StartPipelineRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        match start_pipeline_service(&self.state, request).await {
            Ok(response) => Self::result(response),
            Err(error) => Ok(Self::failure(error)),
        }
    }

    #[tool(
        name = "koharu.list_operations",
        description = "List pipeline operations of this server process with their status and progress"
    )]
    pub(crate) async fn list_operations(&self) -> Result<CallToolResult, ErrorData> {
        // `structuredContent` is an object by MCP definition (strict clients
        // reject a bare array); the HTTP list keeps its bare array shape.
        Self::result(rmcp::serde_json::json!({
            "operations": list_operations_service(&self.state),
        }))
    }

    #[tool(
        name = "koharu.stop_operation",
        description = "Request cancellation of a pipeline operation"
    )]
    pub(crate) async fn stop_operation(
        &self,
        Parameters(request): Parameters<StopOperationRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        stop_operation_service(&self.state, &request.id);
        Ok(CallToolResult::structured(rmcp::serde_json::json!({
            "operationId": request.id,
        })))
    }

    #[tool(
        name = "koharu.export_pages",
        description = "Export rendered pages as PNG or PSD files; returns the written file paths"
    )]
    pub(crate) async fn export_pages(
        &self,
        Parameters(request): Parameters<ExportRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        match export_service(&self.state, request).await {
            Ok(response) => Self::result(response),
            Err(error) => Ok(Self::failure(error)),
        }
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StopOperationRequest {
    #[schemars(description = "id of the operation to cancel")]
    pub(crate) id: String,
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for KoharuTools {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("koharu-rpc", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Koharu is a manga translation pipeline. Open a project, add page images, \
                 run the pipeline with koharu.start_pipeline, poll koharu.list_operations \
                 until the run completes, then export the rendered pages.",
            )
    }
}

/// Mounts the MCP transport on an existing router; requests are served
/// statelessly, each one getting a fresh `KoharuTools` over the shared
/// [`AppState`].
///
/// `json_response` prefers plain JSON replies for request/response calls
/// (the transport falls back to SSE framing automatically if a handler ever
/// emits intermediate notifications), which keeps curl-style clients simple.
pub(crate) fn mount(router: Router, state: AppState, bound_host: IpAddr) -> Router {
    // Host policy: the loopback names plus the bind target, mirroring the
    // CLI's non-loopback warning; `/api/v1` has no equivalent guard.
    let mut allowed_hosts = vec![
        "localhost".to_owned(),
        "127.0.0.1".to_owned(),
        "::1".to_owned(),
    ];
    if !bound_host.is_unspecified() {
        allowed_hosts.push(bound_host.to_string());
    }
    router.nest_service(
        "/mcp",
        StreamableHttpService::new(
            move || Ok(KoharuTools::new(Arc::clone(&state))),
            Arc::new(LocalSessionManager::default()),
            StreamableHttpServerConfig::default()
                .with_allowed_hosts(allowed_hosts)
                .with_legacy_session_mode(false)
                .with_json_response(true),
        ),
    )
}
