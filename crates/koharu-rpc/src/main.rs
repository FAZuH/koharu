use std::net::{IpAddr, Ipv4Addr};

use anyhow::Result;
use axum::{
    Json, Router,
    extract::Request,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use clap::Parser;
use serde::Serialize;
use tokio::net::TcpListener;

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

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();

    if !arguments.host.is_loopback() {
        eprintln!(
            "warning: binding {} exposes the unauthenticated API beyond loopback",
            arguments.host
        );
    }

    let listener = TcpListener::bind((arguments.host, arguments.port))
        .await
        .map_err(|err| {
            if err.kind() == std::io::ErrorKind::AddrInUse {
                anyhow::anyhow!(
                    "port {} is busy on {}; pass --port",
                    arguments.port,
                    arguments.host
                )
            } else {
                err.into()
            }
        })?;
    println!(
        "koharu-rpc listening on http://{}:{}",
        arguments.host, arguments.port
    );

    axum::serve(listener, router())
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install ctrl-c handler");
        })
        .await?;

    Ok(())
}

fn router() -> Router {
    Router::new()
        .route("/api/v1/health", get(health))
        .fallback(not_found)
}

async fn health() -> Json<Health> {
    Json(Health {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

async fn not_found(request: Request) -> Response {
    let status = StatusCode::NOT_FOUND;
    (
        status,
        Json(ApiError {
            status: status.as_u16(),
            message: format!("no route for {}", request.uri().path()),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::router;
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use serde_json::Value;
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_reports_ok_status_and_package_version() {
        let response = router()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["status"], "ok");
        assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn unknown_route_returns_json_error_envelope() {
        let response = router()
            .oneshot(Request::builder().uri("/nope").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            response.headers()["content-type"].to_str().unwrap(),
            "application/json"
        );
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["status"], 404);
        assert_eq!(body["message"], "no route for /nope");
    }
}
