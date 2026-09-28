//! Rendezvous between the translation command and the MCP tool it calls back.
//!
//! The command provider registers a chunk before it spawns anything, so a
//! submission can only ever be matched against a chunk that is actually
//! waiting. The slot is removed as soon as it is answered, which makes a
//! second submission for the same chunk a visible error instead of a silent
//! overwrite.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use tokio::sync::oneshot;

/// One unit of work: the opaque handle handed to the command, and the segment
/// ids the command is required to translate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkRequest {
    pub id: u64,
    pub segments: Vec<(usize, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkAnswer {
    pub translations: Vec<(usize, String)>,
}

struct Slot {
    expected: Vec<usize>,
    send: oneshot::Sender<ChunkAnswer>,
}

/// A synchronous mutex, not the async one: every critical section is a single
/// map operation, and `Drop` cannot await, so a cancelled run has to be able to
/// release its slot without a future.
#[derive(Clone, Default)]
pub struct ChunkSubmissions(Arc<Mutex<HashMap<u64, Slot>>>);

/// Waiting half of a chunk. Dropping it releases the slot, so a cancelled or
/// abandoned chunk can never strand an id that a later run would collide with.
pub struct ChunkTicket {
    submissions: ChunkSubmissions,
    id: u64,
    answer: oneshot::Receiver<ChunkAnswer>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SubmitError {
    #[error("chunk {0} is not waiting for a translation")]
    UnknownChunk(u64),
    #[error("segment {got} is not part of chunk {chunk}; it holds {expected:?}")]
    UnexpectedId {
        chunk: u64,
        got: usize,
        expected: Vec<usize>,
    },
    #[error("segment {0} was submitted more than once for the same chunk")]
    DuplicateId(usize),
    #[error("segment {0} was submitted with empty text")]
    EmptyText(usize),
    #[error("chunk {chunk} was answered with {answered} of {expected} segments")]
    Incomplete {
        chunk: u64,
        answered: usize,
        expected: usize,
    },
}

fn validate(
    chunk: u64,
    expected: &[usize],
    translations: &[(usize, String)],
) -> Result<(), SubmitError> {
    let mut seen: Vec<usize> = Vec::with_capacity(translations.len());
    for (id, text) in translations {
        if !expected.contains(id) {
            return Err(SubmitError::UnexpectedId {
                chunk,
                got: *id,
                expected: expected.to_vec(),
            });
        }
        if seen.contains(id) {
            return Err(SubmitError::DuplicateId(*id));
        }
        if text.trim().is_empty() {
            return Err(SubmitError::EmptyText(*id));
        }
        seen.push(*id);
    }
    if seen.len() != expected.len() {
        return Err(SubmitError::Incomplete {
            chunk,
            answered: seen.len(),
            expected: expected.len(),
        });
    }
    Ok(())
}

impl ChunkSubmissions {
    fn slots(&self) -> std::sync::MutexGuard<'_, HashMap<u64, Slot>> {
        self.0.lock().expect("the submission map is never poisoned")
    }

    /// Reserves `id` for `expected`. A collision means the same chunk id was
    /// issued twice, which would let one command answer another's work.
    pub fn expect(&self, id: u64, expected: Vec<usize>) -> Result<ChunkTicket, u64> {
        let (send, answer) = oneshot::channel();
        let mut slots = self.slots();
        if slots.contains_key(&id) {
            return Err(id);
        }
        slots.insert(id, Slot { expected, send });
        Ok(ChunkTicket {
            submissions: self.clone(),
            id,
            answer,
        })
    }

    /// Validates and delivers. A rejected submission keeps the slot so the
    /// command can correct itself; an accepted one consumes it, so a repeat
    /// call cannot reach a chunk that is already done.
    pub fn submit(
        &self,
        chunk: u64,
        translations: Vec<(usize, String)>,
    ) -> Result<usize, SubmitError> {
        let mut slots = self.slots();
        let Some(slot) = slots.get(&chunk) else {
            return Err(SubmitError::UnknownChunk(chunk));
        };
        validate(chunk, &slot.expected, &translations)?;
        let accepted = translations.len();
        let slot = slots.remove(&chunk).expect("the slot was just found");
        let _ = slot.send.send(ChunkAnswer { translations });
        Ok(accepted)
    }

    pub fn abort(&self, id: u64) {
        self.slots().remove(&id);
    }
}

impl Drop for ChunkTicket {
    fn drop(&mut self) {
        self.submissions.abort(self.id);
    }
}

impl ChunkTicket {
    /// Resolves when the command answers. Passing `deadline` is what bounds a
    /// command that never submits; it may be called again after the process
    /// exits, because a submission can still be in flight at that moment.
    pub async fn answer(&mut self, deadline: Duration) -> Result<ChunkAnswer, &'static str> {
        match tokio::time::timeout(deadline, &mut self.answer).await {
            Ok(Ok(answer)) => Ok(answer),
            Ok(Err(_)) => Err("the chunk was abandoned before it was answered"),
            Err(_) => Err("no translation was submitted"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments(ids: &[usize]) -> Vec<(usize, String)> {
        ids.iter().map(|id| (*id, format!("source {id}"))).collect()
    }

    fn long() -> Duration {
        Duration::from_secs(5)
    }

    #[tokio::test]
    async fn returns_the_submitted_translation() {
        let submissions = ChunkSubmissions::default();
        let mut ticket = submissions.expect(0, vec![0, 1]).unwrap();

        let accepted = submissions
            .submit(0, vec![(0, "hello".to_owned()), (1, "goodbye".to_owned())])
            .unwrap();

        assert_eq!(accepted, 2);
        assert_eq!(
            ticket.answer(long()).await.unwrap(),
            ChunkAnswer {
                translations: vec![(0, "hello".to_owned()), (1, "goodbye".to_owned())]
            }
        );
    }

    #[test]
    fn rejects_a_chunk_that_is_not_waiting() {
        let submissions = ChunkSubmissions::default();

        assert_eq!(
            submissions.submit(7, segments(&[0, 1])),
            Err(SubmitError::UnknownChunk(7))
        );
    }

    #[test]
    fn rejects_a_segment_outside_the_chunk() {
        let submissions = ChunkSubmissions::default();
        let _ticket = submissions.expect(0, vec![0, 1]).unwrap();

        assert_eq!(
            submissions.submit(0, segments(&[0, 9])),
            Err(SubmitError::UnexpectedId {
                chunk: 0,
                got: 9,
                expected: vec![0, 1]
            })
        );
    }

    #[test]
    fn rejects_a_repeated_segment() {
        let submissions = ChunkSubmissions::default();
        let _ticket = submissions.expect(0, vec![0, 1]).unwrap();

        assert_eq!(
            submissions.submit(0, vec![(0, "a".to_owned()), (0, "b".to_owned())]),
            Err(SubmitError::DuplicateId(0))
        );
    }

    #[test]
    fn rejects_blank_text() {
        let submissions = ChunkSubmissions::default();
        let _ticket = submissions.expect(0, vec![0, 1]).unwrap();

        assert_eq!(
            submissions.submit(0, vec![(0, "   ".to_owned()), (1, "b".to_owned())]),
            Err(SubmitError::EmptyText(0))
        );
    }

    #[test]
    fn rejects_a_chunk_missing_segments() {
        let submissions = ChunkSubmissions::default();
        let _ticket = submissions.expect(0, vec![0, 1, 2]).unwrap();

        assert_eq!(
            submissions.submit(0, segments(&[0, 1])),
            Err(SubmitError::Incomplete {
                chunk: 0,
                answered: 2,
                expected: 3
            })
        );
    }

    #[tokio::test]
    async fn a_rejected_submission_leaves_the_chunk_answerable() {
        let submissions = ChunkSubmissions::default();
        let mut ticket = submissions.expect(0, vec![0, 1]).unwrap();
        assert!(submissions.submit(0, segments(&[0, 9])).is_err());

        submissions
            .submit(0, vec![(0, "hello".to_owned()), (1, "goodbye".to_owned())])
            .unwrap();

        assert_eq!(ticket.answer(long()).await.unwrap().translations.len(), 2);
    }

    #[tokio::test]
    async fn a_second_submission_cannot_reach_an_answered_chunk() {
        let submissions = ChunkSubmissions::default();
        let mut ticket = submissions.expect(0, vec![0, 1]).unwrap();
        submissions
            .submit(0, vec![(0, "hello".to_owned()), (1, "goodbye".to_owned())])
            .unwrap();
        ticket.answer(long()).await.unwrap();

        assert_eq!(
            submissions.submit(0, vec![(0, "late".to_owned()), (1, "late".to_owned())]),
            Err(SubmitError::UnknownChunk(0))
        );
    }

    #[test]
    fn registering_the_same_chunk_twice_is_refused() {
        let submissions = ChunkSubmissions::default();
        let _first = submissions.expect(3, vec![0]).unwrap();

        assert_eq!(submissions.expect(3, vec![0]).err(), Some(3));
    }

    #[test]
    fn a_dropped_ticket_releases_its_id() {
        let submissions = ChunkSubmissions::default();
        drop(submissions.expect(0, vec![0]).unwrap());

        // The id is free again, so a cancelled run cannot strand it.
        assert!(submissions.expect(0, vec![0]).is_ok());
    }

    #[tokio::test]
    async fn an_abandoned_chunk_reports_why_it_never_answered() {
        let submissions = ChunkSubmissions::default();
        let mut ticket = submissions.expect(0, vec![0]).unwrap();
        submissions.abort(0);

        assert_eq!(
            ticket.answer(long()).await,
            Err("the chunk was abandoned before it was answered")
        );
    }

    #[tokio::test]
    async fn a_waiting_chunk_times_out_when_nothing_is_submitted() {
        let submissions = ChunkSubmissions::default();
        let mut ticket = submissions.expect(0, vec![0]).unwrap();

        assert_eq!(
            ticket.answer(Duration::from_millis(20)).await,
            Err("no translation was submitted")
        );
    }
}
