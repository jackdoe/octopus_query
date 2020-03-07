const NO_MORE: i32 = std::i32::MAX;
const NOT_READY: i32 = -1;

/// idf is log(1 + N/D)
/// ```
/// N = total documents in the index
/// d = documents matching (len(postings))
/// https://en.wikipedia.org/wiki/tf-idf
/// ```
pub fn compute_idf(n: usize, d: usize) -> f32 {
    let nf = n as f32;
    let df = d as f32;
    let x = nf / df;
    return x.ln_1p();
}
/// Basic term query
pub struct Term<'a> {
    cursor: usize,
    idf: f32,
    doc_id: i32,
    postings: &'a [i32],
}

impl<'a> Term<'a> {
    /// Create Term query
    ///
    /// Arguments
    ///
    /// * `n_docs_in_index` - Total documents in your index, in order to compute idf, pass 1 if you dont care
    /// * `postings` - sorted array of integer document ids
    pub fn new(n_docs_in_index: usize, postings: &'a [i32]) -> Self {
        let d = postings.len();
        Self {
            postings: postings,
            doc_id: NOT_READY,
            cursor: 0,
            idf: compute_idf(n_docs_in_index, d),
        }
    }
}

impl<'a> Query for Term<'a> {
    fn advance(&mut self, target: i32) -> i32 {
        let mut start = self.cursor;
        let mut end = self.postings.len();

        while start < end {
            let mid = start + ((end - start) >> 1);
            let current = self.postings[mid];
            if current == target {
                self.cursor = mid;
                self.doc_id = target;
                return target;
            }

            if current < target {
                start = mid + 1;
            } else {
                end = mid;
            }
        }

        if start >= self.postings.len() {
            self.doc_id = NO_MORE;
            return NO_MORE;
        }

        self.cursor = start;
        self.doc_id = self.postings[start];
        return self.doc_id;
    }

    fn next(&mut self) -> i32 {
        if self.doc_id != NOT_READY {
            self.cursor += 1;
        }

        if self.cursor >= self.postings.len() {
            self.doc_id = NO_MORE
        } else {
            self.doc_id = self.postings[self.cursor]
        }
        return self.doc_id;
    }

    fn doc_id(&self) -> i32 {
        return self.doc_id;
    }

    fn score(&self) -> f32 {
        return self.idf;
    }
}
/// BooleanAnd query
pub struct And<'a> {
    doc_id: i32,
    queries: &'a mut [&'a mut dyn Query],
}

impl<'a> And<'a> {
    /// Create BooleanAnd query
    ///
    /// Arguments
    ///
    /// * `queries` - subqueries (same as lucene's Boolean MUST clause)
    pub fn new(queries: &'a mut [&'a mut dyn Query]) -> Self {
        Self {
            doc_id: NOT_READY,
            queries: queries,
        }
    }
    fn next_anded_doc(&mut self, mut target: i32) -> i32 {
        let mut i: usize = 1;
        while i < self.queries.len() {
            let q = &mut self.queries[i];
            let mut q_doc_id = q.doc_id();
            if q_doc_id < target {
                q_doc_id = q.advance(target)
            }
            if q_doc_id == target {
                i = i + 1;
                continue;
            }
            target = self.queries[0].advance(q_doc_id);
            i = 0
        }
        self.doc_id = target;
        return self.doc_id;
    }
}

impl<'a> Query for And<'a> {
    fn advance(&mut self, target: i32) -> i32 {
        if self.queries.len() == 0 {
            return NO_MORE;
        }
        let t = self.queries[0].advance(target);
        return self.next_anded_doc(t);
    }

    fn next(&mut self) -> i32 {
        if self.queries.len() == 0 {
            return NO_MORE;
        }
        let t = self.queries[0].next();
        return self.next_anded_doc(t);
    }

    fn doc_id(&self) -> i32 {
        return self.doc_id;
    }

    fn score(&self) -> f32 {
        self.queries
            .iter()
            .filter_map(|q| {
                if q.doc_id() == self.doc_id {
                    Some(q.score())
                } else {
                    None
                }
            })
            .sum()
    }
}

/// BooleanOr query
pub struct Or<'a> {
    doc_id: i32,
    queries: &'a mut [&'a mut dyn Query],
}

impl<'a> Or<'a> {
    /// Create BooleanOr query
    ///
    /// Arguments
    ///
    /// * `queries` - subqueries (same as lucene's Boolean Should clause)
    pub fn new(queries: &'a mut [&'a mut dyn Query]) -> Self {
        Self {
            doc_id: NOT_READY,
            queries: queries,
        }
    }
}

impl<'a> Query for Or<'a> {
    fn advance(&mut self, target: i32) -> i32 {
        let mut new_doc_id: i32 = NO_MORE;
        let mut i: usize = 0;
        while i < self.queries.len() {
            let q = &mut self.queries[i];
            let mut cur_doc_id = q.doc_id();
            if cur_doc_id < target {
                cur_doc_id = q.advance(target)
            }

            if cur_doc_id < new_doc_id {
                new_doc_id = cur_doc_id
            }
            i += 1;
        }
        self.doc_id = new_doc_id;
        return self.doc_id;
    }

    fn next(&mut self) -> i32 {
        let mut new_doc_id: i32 = NO_MORE;
        let mut i: usize = 0;
        while i < self.queries.len() {
            let q = &mut self.queries[i];
            let mut cur_doc_id = q.doc_id();
            if cur_doc_id == self.doc_id {
                cur_doc_id = q.next()
            }

            if cur_doc_id < new_doc_id {
                new_doc_id = cur_doc_id
            }
            i += 1;
        }
        self.doc_id = new_doc_id;
        return self.doc_id;
    }

    fn doc_id(&self) -> i32 {
        return self.doc_id;
    }

    fn score(&self) -> f32 {
        self.queries
            .iter()
            .map(|q| {
                if q.doc_id() == self.doc_id {
                    q.score()
                } else {
                    0.0
                }
            })
            .sum()
    }
}

/// DisMax query
pub struct DisMax<'a> {
    doc_id: i32,
    tiebreaker: f32,
    queries: &'a mut [&'a mut dyn Query],
}

impl<'a> DisMax<'a> {
    /// Create DisMax query
    /// picks the max score, and then it adds tiebreaker * score for the other matching queries
    ///
    /// Arguments
    ///
    /// * `tiebreaker` - the tiebreaker factor
    /// * `queries` - subqueries
    pub fn new(tiebreaker: f32, queries: &'a mut [&'a mut dyn Query]) -> Self {
        Self {
            doc_id: NOT_READY,
            tiebreaker: tiebreaker,
            queries: queries,
        }
    }
}

impl<'a> Query for DisMax<'a> {
    fn advance(&mut self, target: i32) -> i32 {
        let mut new_doc_id: i32 = NO_MORE;
        let mut i: usize = 0;
        while i < self.queries.len() {
            let q = &mut self.queries[i];
            let mut cur_doc_id = q.doc_id();
            if cur_doc_id < target {
                cur_doc_id = q.advance(target)
            }

            if cur_doc_id < new_doc_id {
                new_doc_id = cur_doc_id
            }
            i += 1;
        }
        self.doc_id = new_doc_id;
        return self.doc_id;
    }

    fn next(&mut self) -> i32 {
        let mut new_doc_id: i32 = NO_MORE;
        let mut i: usize = 0;
        while i < self.queries.len() {
            let q = &mut self.queries[i];
            let mut cur_doc_id = q.doc_id();
            if cur_doc_id == self.doc_id {
                cur_doc_id = q.next()
            }

            if cur_doc_id < new_doc_id {
                new_doc_id = cur_doc_id
            }
            i += 1;
        }
        self.doc_id = new_doc_id;
        return self.doc_id;
    }

    fn doc_id(&self) -> i32 {
        return self.doc_id;
    }

    fn score(&self) -> f32 {
        let mut sum: f32 = 0.0;
        let mut max: f32 = 0.0;
        for q in self.queries.iter() {
            if q.doc_id() == self.doc_id {
                let score = q.score();
                if score > max {
                    max = score
                }
                sum += score;
            }
        }

        max + (sum - max) * self.tiebreaker
    }
}

/// Constant Score query
pub struct Constant<'a> {
    boost: f32,
    query: &'a mut dyn Query,
}

impl<'a> Constant<'a> {
    /// Create Constant Score query
    ///
    /// ignores the subquery score and returns `boost` score
    ///
    /// Arguments
    ///
    /// * `boost` - the score being returned
    /// * `query` - the sub query
    pub fn new(boost: f32, query: &'a mut dyn Query) -> Self {
        Self {
            boost: boost,
            query: query,
        }
    }
}

impl<'a> Query for Constant<'a> {
    fn advance(&mut self, target: i32) -> i32 {
        return self.query.advance(target);
    }

    fn next(&mut self) -> i32 {
        return self.query.next();
    }

    fn doc_id(&self) -> i32 {
        return self.query.doc_id();
    }

    fn score(&self) -> f32 {
        return self.boost;
    }
}

pub trait Query {
    fn advance(&mut self, target: i32) -> i32;
    fn next(&mut self) -> i32;
    fn doc_id(&self) -> i32;
    fn score(&self) -> f32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::*;

    #[test]
    fn test_term_next() {
        let mut t = Term::new(1, &[1, 2, 3]);
        assert_eq!(t.next(), 1);
        assert_eq!(t.next(), 2);
        assert_eq!(t.next(), 3);
        assert_eq!(t.next(), NO_MORE);
    }

    #[test]
    fn test_term_advance() {
        let mut t = Term::new(1, &[1, 2, 3, 5]);
        assert_eq!(t.advance(1), 1);
        assert_eq!(t.advance(4), 5);
        assert_eq!(t.advance(5), 5);
        assert_eq!(t.advance(6), NO_MORE);
    }

    #[test]
    fn test_and_advance() {
        let queries: &mut [&mut dyn Query] = &mut [
            &mut Term::new(1, &[1, 2, 3, 5, 6]),
            &mut Term::new(1, &[1, 2, 4, 5, 6]),
        ];
        let mut and = And::new(queries);
        assert_eq!(and.advance(4), 5);
        assert_eq!(and.next(), 6);
        assert_eq!(and.next(), NO_MORE);
    }

    #[test]
    fn test_and_next() {
        let queries: &mut [&mut dyn Query] = &mut [
            &mut Term::new(1, &[1, 2, 3]),
            &mut Term::new(1, &[1, 2, 4, 5]),
            &mut Term::new(1, &[1, 2, 3]),
            &mut Term::new(1, &[1, 2, 7]),
        ];
        let mut and = And::new(queries);
        assert_eq!(and.next(), 1);
        assert_eq!(and.next(), 2);
        assert_eq!(and.next(), NO_MORE);
    }

    #[test]
    fn test_and_empty() {
        let mut and = Or::new(&mut []);
        assert_eq!(and.next(), NO_MORE);
        assert_eq!(and.advance(1), NO_MORE);
    }

    #[test]
    fn test_or_next() {
        let queries: &mut [&mut dyn Query] = &mut [
            &mut Term::new(1, &[1, 2, 3]),
            &mut Term::new(1, &[1, 2, 4, 5]),
        ];
        let mut or = Or::new(queries);
        assert_eq!(or.next(), 1);
        assert_eq!(or.score(), compute_idf(1, 3) + compute_idf(1, 4));

        assert_eq!(or.next(), 2);
        assert_eq!(or.score(), compute_idf(1, 3) + compute_idf(1, 4));

        assert_eq!(or.next(), 3);
        assert_eq!(or.score(), compute_idf(1, 3));

        assert_eq!(or.next(), 4);
        assert_eq!(or.score(), compute_idf(1, 4));

        assert_eq!(or.next(), 5);
        assert_eq!(or.next(), NO_MORE);
    }

    #[test]
    fn test_or_advance() {
        let queries: &mut [&mut dyn Query] = &mut [
            &mut Term::new(1, &[1, 2, 3]),
            &mut Term::new(1, &[1, 2, 4, 5]),
        ];
        let mut or = Or::new(queries);
        assert_eq!(or.advance(4), 4);
        assert_eq!(or.next(), 5);
        assert_eq!(or.next(), NO_MORE);
    }

    #[test]
    fn test_or_empty() {
        let mut or = Or::new(&mut []);
        assert_eq!(or.next(), NO_MORE);
        assert_eq!(or.advance(1), NO_MORE);
    }

    #[test]
    fn test_or_complex() {
        let queries: &mut [&mut dyn Query] =
            &mut [&mut Term::new(1, &[1, 2, 3]), &mut Term::new(1, &[1, 7, 9])];
        let mut or = Or::new(queries);

        let queries: &mut [&mut dyn Query] = &mut [
            &mut Term::new(1, &[1, 2, 7]),
            &mut Term::new(1, &[1, 2, 4, 5, 7, 9]),
            &mut or,
        ];
        let mut and = And::new(queries);

        assert_eq!(and.next(), 1);
        assert!(nearly_equal(
            and.score(),
            compute_idf(1, 3) + compute_idf(1, 3) + compute_idf(1, 3) + compute_idf(1, 6)
        ));

        assert_eq!(and.next(), 2);
        assert_eq!(and.score(), 0.72951484);

        assert_eq!(and.next(), 7);
        assert_eq!(and.score(), 0.72951484);

        assert_eq!(and.next(), NO_MORE);
    }

    #[test]
    fn test_example() {
        let queries: &mut [&mut dyn Query] =
            &mut [&mut Term::new(1, &[1, 2, 3]), &mut Term::new(1, &[1, 7, 9])];
        let mut or = Or::new(queries);

        let queries: &mut [&mut dyn Query] = &mut [
            &mut Term::new(1, &[1, 2, 7]),
            &mut Term::new(1, &[1, 2, 4, 5, 7, 9]),
            &mut or,
        ];
        let mut and = And::new(queries);

        while and.next() != NO_MORE {
            println!("doc: {}, score: {}", and.doc_id(), and.score());
        }
    }

    pub fn nearly_equal(a: f32, b: f32) -> bool {
        let abs_a = a.abs();
        let abs_b = b.abs();
        let diff = (a - b).abs();

        if a == b {
            // Handle infinities.
            true
        } else if a == 0.0 || b == 0.0 || diff < f32::MIN_POSITIVE {
            // One of a or b is zero (or both are extremely close to it,) use absolute error.
            diff < (f32::EPSILON * f32::MIN_POSITIVE)
        } else {
            // Use relative error.
            (diff / f32::min(abs_a + abs_b, f32::MAX)) < f32::EPSILON
        }
    }

    #[test]
    fn test_constant_next() {
        let queries: &mut [&mut dyn Query] = &mut [
            &mut Term::new(1, &[1, 2, 3]),
            &mut Term::new(1, &[1, 2, 4, 5]),
        ];
        let or: &mut dyn Query = &mut Or::new(queries);
        let mut c = Constant::new(1.3, or);
        assert_eq!(c.next(), 1);
        assert_eq!(c.score(), 1.3);

        assert_eq!(c.next(), 2);
        assert_eq!(c.score(), 1.3);

        assert_eq!(c.next(), 3);
        assert_eq!(c.score(), 1.3);

        assert_eq!(c.next(), 4);
        assert_eq!(c.score(), 1.3);

        assert_eq!(c.next(), 5);
        assert_eq!(c.next(), NO_MORE);
    }

    #[test]
    fn test_dismax_next() {
        let queries_a: &mut [&mut dyn Query] = &mut [
            &mut Term::new(1, &[1, 2, 3]),
            &mut Term::new(1, &[1, 2, 4, 5]),
        ];
        let queries_b: &mut [&mut dyn Query] = &mut [
            &mut Term::new(1, &[1, 2, 3]),
            &mut Term::new(1, &[1, 2, 4, 5]),
        ];

        let or_a: &mut dyn Query = &mut Or::new(queries_a);
        let or_b: &mut dyn Query = &mut Or::new(queries_b);
        let mut ca = Constant::new(1.3, or_a);
        let mut cb = Constant::new(1.3, or_b);

        let queries: &mut [&mut dyn Query] = &mut [&mut ca, &mut cb];
        let mut dm = DisMax::new(0.1, queries);
        assert_eq!(dm.next(), 1);
        assert_eq!(dm.score(), 1.43);
    }
}
