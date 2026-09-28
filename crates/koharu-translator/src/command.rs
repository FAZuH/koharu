//! Translation delegated to an external command, one process per chunk.
//!
//! The command receives the source segments as text and is expected to answer
//! by calling the `submit_translations` MCP tool, so the result arrives as
//! typed arguments rather than as prose the pipeline would have to parse out
//! of a model response. Nothing about the command is assumed beyond that: it
//! may be an agent harness, a script, or a test double.

use std::{
    path::PathBuf,
    process::Stdio,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use tokio::io::AsyncReadExt;

use super::submissions::{ChunkAnswer, ChunkRequest, ChunkSubmissions};
use crate::{Result, TranslationRequest, error::Error, prompt};

/// Placeholder for the chunk handle the command must echo back.
const ID_PLACEHOLDER: &str = "{{id}}";
/// Placeholder for the source segments to translate.
const TEXT_PLACEHOLDER: &str = "{{text}}";
/// stderr past this length is dropped; the tail is what explains a failure.
const STDERR_TAIL: usize = 2048;

/// How long a submission may still arrive after its command has exited. The
/// call is an in-flight HTTP request at that point, so this covers the race
/// rather than any real work — waiting the whole budget here would stall every
/// command that gives up early.
const SUBMIT_GRACE: Duration = Duration::from_secs(2);

/// The slot map is shared by every run in this process, so chunk ids come from
/// here rather than from a per-page counter.
static NEXT_CHUNK: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct CommandConfig {
    /// Argument vector, with `{{id}}` and `{{text}}` substituted per chunk.
    /// Rendered without a shell, so a placeholder may expand to text
    /// containing quotes or newlines without changing the command.
    pub argv: Vec<String>,
    /// Working directory for the command, which is also where a project-scoped
    /// agent configuration such as the MCP server registration is resolved.
    pub cwd: Option<PathBuf>,
    /// Budget for one chunk, covering both the process and its submission.
    #[specta(type = f64)]
    pub timeout_secs: u64,
    /// Segments per chunk. Larger chunks mean fewer processes but a single bad
    /// page loses the whole chunk.
    #[specta(type = f64)]
    pub segments_per_chunk: usize,
}

impl Default for CommandConfig {
    fn default() -> Self {
        Self {
            argv: Vec::new(),
            cwd: None,
            timeout_secs: 180,
            segments_per_chunk: 8,
        }
    }
}

pub(super) async fn translate(
    config: &CommandConfig,
    submissions: Option<&ChunkSubmissions>,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let Some(submissions) = submissions else {
        return Err(anyhow::anyhow!(
            "the command provider needs a submission channel, which only the server transport has"
        )
        .into());
    };
    if config.segments_per_chunk == 0 {
        return Err(
            anyhow::anyhow!("command provider segments_per_chunk must be at least 1").into(),
        );
    }

    let mut translated: Vec<Option<String>> = vec![None; request.segments.len()];
    for mut request in chunk(&request.segments, config.segments_per_chunk) {
        // Slot ids are process-wide, not page-relative: the map is shared by
        // every run in this process, so a per-run counter would let a second
        // run collide with the first one's chunk 0.
        let id = NEXT_CHUNK.fetch_add(1, Ordering::Relaxed);
        let expected: Vec<usize> = request
            .segments
            .iter()
            .map(|(segment, _)| *segment)
            .collect();
        let ticket = submissions
            .expect(id, expected)
            .map_err(|id| anyhow::anyhow!("chunk {id} is already in flight"))?;
        request.id = id;
        let answer = run(config, &request, ticket).await?;
        for (segment, text) in answer.translations {
            translated[segment] = Some(text);
        }
    }
    // Every chunk is validated complete before it resolves, so the only way to
    // reach here with a hole is a segment no chunk owned.
    translated
        .into_iter()
        .enumerate()
        .map(|(id, text)| {
            text.ok_or_else(|| Error::Other(anyhow::anyhow!("segment {id} was never translated")))
        })
        .collect::<Result<Vec<String>>>()
}

fn chunk(segments: &[String], per_chunk: usize) -> Vec<ChunkRequest> {
    segments
        .chunks(per_chunk)
        .enumerate()
        .map(|(index, group)| ChunkRequest {
            id: index as u64,
            segments: group
                .iter()
                .enumerate()
                .map(|(offset, text)| (index * per_chunk + offset, text.clone()))
                .collect(),
        })
        .collect()
}

async fn run(
    config: &CommandConfig,
    request: &ChunkRequest,
    ticket: super::submissions::ChunkTicket,
) -> Result<super::submissions::ChunkAnswer> {
    let budget = Duration::from_secs(config.timeout_secs);
    let deadline = tokio::time::Instant::now() + budget;
    let (program, args) = render(config, request)?;
    let mut child = tokio::process::Command::new(program);
    if let Some(cwd) = config.cwd.as_deref() {
        child.current_dir(cwd);
    }
    let mut child = child
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| anyhow::anyhow!("failed to start the command: {error}"))?;

    // Draining stderr concurrently keeps a chatty command from filling the
    // pipe buffer and deadlocking against its own wait.
    let mut stderr = child.stderr.take().expect("stderr was piped");
    let drain = tokio::spawn(async move {
        let mut buffer = Vec::new();
        let _ = stderr.read_to_end(&mut buffer).await;
        buffer
    });

    // A command that submits and then lingers has still delivered a usable
    // answer, so the submission races the process rather than following it. The
    // answer arm owns the deadline, which is what bounds a command that neither
    // submits nor exits.
    let mut ticket = ticket;
    let remaining = || deadline.saturating_duration_since(tokio::time::Instant::now());
    let outcome = tokio::select! {
        answer = ticket.answer(remaining()) => Outcome::Answer(answer),
        status = child.wait() => Outcome::Exited(status),
    };
    match outcome {
        Outcome::Answer(Ok(answer)) => {
            let _ = child.start_kill();
            Ok(answer)
        }
        Outcome::Answer(Err(reason)) => {
            let _ = child.start_kill();
            let detail = tail(&drain.await.unwrap_or_default());
            Err(anyhow::anyhow!(
                "the command neither submitted nor exited within {}s for chunk {}: \
                 {reason}{detail}",
                config.timeout_secs,
                request.id
            )
            .into())
        }
        Outcome::Exited(Err(error)) => Err(anyhow::anyhow!("the command failed: {error}").into()),
        Outcome::Exited(Ok(status)) => match ticket.answer(remaining().min(SUBMIT_GRACE)).await {
            // The submission can still be in flight as the process exits.
            Ok(answer) => Ok(answer),
            Err(reason) => {
                let how = if status.success() {
                    "exited successfully without calling submit_translations".to_owned()
                } else {
                    format!("exited with {status}")
                };
                let detail = tail(&drain.await.unwrap_or_default());
                Err(anyhow::anyhow!(
                    "the command {how} for chunk {}: {reason}{detail}",
                    request.id
                )
                .into())
            }
        },
    }
}

enum Outcome {
    Answer(std::result::Result<ChunkAnswer, &'static str>),
    Exited(std::io::Result<std::process::ExitStatus>),
}

fn render(config: &CommandConfig, request: &ChunkRequest) -> Result<(String, Vec<String>)> {
    let text = prompt::segments_json(
        &request
            .segments
            .iter()
            .map(|(_, text)| text.clone())
            .collect::<Vec<String>>(),
    );
    let render = |value: &str| {
        value
            .replace(ID_PLACEHOLDER, &request.id.to_string())
            .replace(TEXT_PLACEHOLDER, &text)
    };
    let mut args = config.argv.iter().map(|arg| render(arg));
    let program = args
        .next()
        .ok_or_else(|| Error::Other(anyhow::anyhow!("command provider has no argv")))?;
    Ok((program, args.collect()))
}

fn tail(stderr: &[u8]) -> String {
    if stderr.is_empty() {
        return String::new();
    }
    let start = stderr.len().saturating_sub(STDERR_TAIL);
    let text = String::from_utf8_lossy(&stderr[start..]);
    let text = text.trim();
    if text.is_empty() {
        String::new()
    } else {
        format!("\nstderr: {text}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::submissions::ChunkTicket;

    fn segments(count: usize) -> Vec<String> {
        (0..count).map(|id| format!("source {id}")).collect()
    }

    #[derive(Debug, PartialEq, Eq, serde::Deserialize)]
    struct RenderedSegment {
        id: usize,
        text: String,
    }

    fn request(count: usize) -> ChunkRequest {
        ChunkRequest {
            id: 3,
            segments: (0..count).map(|id| (id, format!("source {id}"))).collect(),
        }
    }

    #[test]
    fn chunks_keep_source_ids_across_the_boundary() {
        let chunks = chunk(&segments(5), 2);

        let ids: Vec<Vec<usize>> = chunks
            .iter()
            .map(|chunk| chunk.segments.iter().map(|(id, _)| *id).collect())
            .collect();
        assert_eq!(ids, vec![vec![0, 1], vec![2, 3], vec![4]]);
        assert_eq!(
            chunks.iter().map(|chunk| chunk.id).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn a_short_final_chunk_is_kept() {
        assert_eq!(chunk(&segments(4), 4).len(), 1);
    }

    #[test]
    fn substitutes_the_chunk_id_and_the_source_text() {
        let config = CommandConfig {
            argv: vec![
                "run".to_owned(),
                "translate chunk {{id}}".to_owned(),
                "{{text}}".to_owned(),
            ],
            ..CommandConfig::default()
        };

        let (program, args) = render(&config, &request(2)).unwrap();

        assert_eq!(program, "run");
        assert_eq!(
            args,
            vec![
                "translate chunk 3".to_owned(),
                r#"[{"id":0,"text":"source 0"},{"id":1,"text":"source 1"}]"#.to_owned(),
            ]
        );
    }

    #[test]
    fn awkward_text_reaches_the_command_intact() {
        let config = CommandConfig {
            argv: vec!["run".to_owned(), "{{text}}".to_owned()],
            ..CommandConfig::default()
        };
        // Quotes, a backslash, a newline and an em dash are what a shell would
        // mangle. The argument is JSON, so decoding it back to the original text
        // is what proves the round trip lost nothing.
        let awkward = "「やあ」\nshe said \"no\" \\ twice\t— end".to_owned();

        let (_, args) = render(
            &config,
            &ChunkRequest {
                id: 0,
                segments: vec![(0, awkward.clone())],
            },
        )
        .unwrap();
        let segments: Vec<RenderedSegment> = serde_json::from_str(&args[0]).unwrap();

        assert_eq!(
            segments,
            vec![RenderedSegment {
                id: 0,
                text: awkward
            }]
        );
    }

    #[test]
    fn every_placeholder_is_expanded_in_every_argument() {
        let config = CommandConfig {
            argv: vec![
                "agent".to_owned(),
                "{{id}}".to_owned(),
                "{{id}}-{{text}}-{{id}}".to_owned(),
            ],
            ..CommandConfig::default()
        };

        let (program, args) = render(&config, &request(1)).unwrap();

        assert_eq!(program, "agent");
        assert_eq!(args[0], "3");
        assert!(!args[1].contains("{{id}}"));
        assert!(!args[1].contains("{{text}}"));
        assert!(args[1].starts_with("3-") && args[1].ends_with("-3"));
    }

    #[test]
    fn stderr_tail_keeps_the_end_of_a_long_stream() {
        let noise = "x".repeat(STDERR_TAIL * 2);

        let detail = tail(noise.as_bytes());

        assert!(detail.starts_with("\nstderr: xxx"));
        assert_eq!(detail.len(), STDERR_TAIL + "\nstderr: ".len());
    }

    #[test]
    fn an_empty_stderr_adds_nothing_to_the_error() {
        assert_eq!(tail(b""), "");
        assert_eq!(tail(b"   \n"), "");
    }

    // The tests below spawn real processes: they are integration tests of the
    // process boundary, which no amount of pure-function testing reaches.

    fn waiting_chunk(id: u64) -> (ChunkSubmissions, ChunkTicket) {
        let submissions = ChunkSubmissions::default();
        let ticket = submissions.expect(id, vec![0]).unwrap();
        (submissions, ticket)
    }

    #[tokio::test]
    async fn a_submission_wins_over_a_command_that_is_still_running() {
        let (submissions, ticket) = waiting_chunk(5);
        let request = ChunkRequest {
            id: 5,
            segments: vec![(0, "やあ".to_owned())],
        };
        // The command submits and then keeps running, which is the case where
        // waiting for the process first would throw the answer away.
        let answerer = tokio::spawn({
            let submissions = submissions.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(150)).await;
                submissions
                    .submit(5, vec![(0, "hello".to_owned())])
                    .unwrap();
            }
        });
        let config = CommandConfig {
            argv: vec!["sleep".to_owned(), "30".to_owned()],
            timeout_secs: 60,
            ..CommandConfig::default()
        };

        let answer = run(&config, &request, ticket).await.unwrap();

        answerer.await.unwrap();
        assert_eq!(answer.translations, vec![(0, "hello".to_owned())]);
    }

    #[tokio::test]
    async fn a_failing_command_reports_its_exit_status_and_stderr() {
        let (_, ticket) = waiting_chunk(0);
        let request = ChunkRequest {
            id: 0,
            segments: vec![(0, "やあ".to_owned())],
        };
        let config = CommandConfig {
            argv: vec![
                "sh".to_owned(),
                "-c".to_owned(),
                "echo out-of-credit >&2; exit 3".to_owned(),
            ],
            timeout_secs: 30,
            ..CommandConfig::default()
        };

        let error = run(&config, &request, ticket)
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("exited with"), "{error}");
        assert!(error.contains("out-of-credit"), "{error}");
    }

    #[tokio::test]
    async fn a_command_that_never_submits_is_reported_with_its_stderr() {
        let (_, ticket) = waiting_chunk(0);
        let request = ChunkRequest {
            id: 0,
            segments: vec![(0, "やあ".to_owned())],
        };
        let config = CommandConfig {
            argv: vec![
                "sh".to_owned(),
                "-c".to_owned(),
                "echo gave-up >&2".to_owned(),
            ],
            // Far larger than the wait this must actually take.
            timeout_secs: 120,
            ..CommandConfig::default()
        };

        let started = tokio::time::Instant::now();
        let error = run(&config, &request, ticket)
            .await
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("without calling submit_translations"),
            "{error}"
        );
        assert!(error.contains("gave-up"), "{error}");
        assert!(
            started.elapsed() < SUBMIT_GRACE * 2,
            "{:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn a_command_that_hangs_is_killed_at_the_deadline() {
        let (_, ticket) = waiting_chunk(0);
        let request = ChunkRequest {
            id: 0,
            segments: vec![(0, "やあ".to_owned())],
        };
        let config = CommandConfig {
            argv: vec!["sleep".to_owned(), "120".to_owned()],
            timeout_secs: 1,
            ..CommandConfig::default()
        };

        let started = tokio::time::Instant::now();
        let error = run(&config, &request, ticket)
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("neither submitted nor exited"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(30));
    }

    #[tokio::test]
    async fn a_command_that_cannot_be_started_fails_immediately() {
        let (_, ticket) = waiting_chunk(0);
        let request = ChunkRequest {
            id: 0,
            segments: vec![(0, "やあ".to_owned())],
        };
        let config = CommandConfig {
            argv: vec!["koharu-no-such-command".to_owned()],
            ..CommandConfig::default()
        };

        let error = run(&config, &request, ticket)
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("failed to start the command"), "{error}");
    }
}
