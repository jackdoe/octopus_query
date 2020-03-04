const NO_MORE: i32 = std::i32::MAX;
const NOT_READY: i32 = -1;

pub struct Term {
    cursor: usize,
    doc_id: i32,
    postings: Vec<i32>,
}
impl Term {
    fn new(postings: Vec<i32>) -> Self {
        Self {
            postings: postings,
            doc_id: NOT_READY,
            cursor: 0,
        }
    }
}

impl Query for Term {
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

    fn cost(&self) -> usize {
        return self.postings.len();
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
        return 1.0;
    }
}

pub struct And {
    doc_id: i32,
    queries: Vec<Box<dyn Query>>,
}

impl And {
    fn new(queries: Vec<Box<dyn Query>>) -> Self {
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

impl Query for And {
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

    fn cost(&self) -> usize {
        let mut cost: usize = 0;
        for q in &self.queries {
            cost += q.cost()
        }
        return cost;
    }

    fn doc_id(&self) -> i32 {
        return self.doc_id;
    }

    fn score(&self) -> f32 {
        let mut score: f32 = 0.0;
        for q in &self.queries {
            if q.doc_id() == self.doc_id {
                score += q.score()
            }
        }
        return score;
    }
}

pub struct Or {
    doc_id: i32,
    queries: Vec<Box<dyn Query>>,
}

impl Or {
    fn new(queries: Vec<Box<dyn Query>>) -> Self {
        Self {
            doc_id: NOT_READY,
            queries: queries,
        }
    }
}

impl Query for Or {
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

    fn cost(&self) -> usize {
        let mut cost: usize = 0;
        for q in &self.queries {
            cost += q.cost()
        }
        return cost;
    }

    fn doc_id(&self) -> i32 {
        return self.doc_id;
    }

    fn score(&self) -> f32 {
        let mut score: f32 = 0.0;
        for q in &self.queries {
            if q.doc_id() == self.doc_id {
                score += q.score()
            }
        }
        return score;
    }
}

pub trait Query {
    fn advance(&mut self, target: i32) -> i32;
    fn next(&mut self) -> i32;
    fn doc_id(&self) -> i32;
    fn score(&self) -> f32;
    fn cost(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_next() {
        let mut t = Term::new([1, 2, 3].to_vec());
        assert_eq!(t.next(), 1);
        assert_eq!(t.next(), 2);
        assert_eq!(t.next(), 3);
        assert_eq!(t.next(), NO_MORE);
    }

    #[test]
    fn test_term_advance() {
        let mut t = Term::new([1, 2, 3, 5].to_vec());
        assert_eq!(t.advance(1), 1);
        assert_eq!(t.advance(4), 5);
        assert_eq!(t.advance(5), 5);
        assert_eq!(t.advance(6), NO_MORE);
    }

    #[test]
    fn test_and_advance() {
        let mut and = And::new(vec![
            Box::new(Term::new([1, 2, 3, 5, 6].to_vec())),
            Box::new(Term::new([1, 2, 4, 5, 6].to_vec())),
        ]);
        assert_eq!(and.advance(4), 5);
        assert_eq!(and.next(), 6);
        assert_eq!(and.next(), NO_MORE);
    }

    #[test]
    fn test_and_next() {
        let mut and = And::new(vec![
            Box::new(Term::new([1, 2, 3].to_vec())),
            Box::new(Term::new([1, 2, 4, 5].to_vec())),
            Box::new(Term::new([1, 2, 3].to_vec())),
            Box::new(Term::new([1, 2, 7].to_vec())),
        ]);
        assert_eq!(and.next(), 1);
        assert_eq!(and.next(), 2);
        assert_eq!(and.next(), NO_MORE);
    }

    #[test]
    fn test_and_empty() {
        let mut and = Or::new(vec![]);
        assert_eq!(and.next(), NO_MORE);
        assert_eq!(and.advance(1), NO_MORE);
    }

    #[test]
    fn test_or_next() {
        let mut or = Or::new(vec![
            Box::new(Term::new([1, 2, 3].to_vec())),
            Box::new(Term::new([1, 2, 4, 5].to_vec())),
        ]);
        assert_eq!(or.next(), 1);
        assert_eq!(or.score(), 2.0);

        assert_eq!(or.next(), 2);
        assert_eq!(or.score(), 2.0);

        assert_eq!(or.next(), 3);
        assert_eq!(or.score(), 1.0);

        assert_eq!(or.next(), 4);
        assert_eq!(or.next(), 5);
        assert_eq!(or.next(), NO_MORE);
    }

    #[test]
    fn test_or_advance() {
        let mut or = Or::new(vec![
            Box::new(Term::new([1, 2, 3].to_vec())),
            Box::new(Term::new([1, 2, 4, 5].to_vec())),
        ]);
        assert_eq!(or.advance(4), 4);
        assert_eq!(or.next(), 5);
        assert_eq!(or.next(), NO_MORE);
    }

    #[test]
    fn test_or_empty() {
        let mut or = Or::new(vec![]);
        assert_eq!(or.next(), NO_MORE);
        assert_eq!(or.advance(1), NO_MORE);
    }

    #[test]
    fn test_or_complex() {
        let or = Or::new(vec![
            Box::new(Term::new([1, 2, 3].to_vec())),
            Box::new(Term::new([1, 7, 9].to_vec())),
        ]);

        let mut and = And::new(vec![
            Box::new(Term::new([1, 2, 7].to_vec())),
            Box::new(Term::new([1, 2, 4, 5, 7, 9].to_vec())),
            Box::new(or),
        ]);

        assert_eq!(and.next(), 1);
        assert_eq!(and.score(), 4.0);

        assert_eq!(and.next(), 2);
        assert_eq!(and.score(), 3.0);

        assert_eq!(and.next(), 7);
        assert_eq!(and.score(), 3.0);

        assert_eq!(and.next(), NO_MORE);
    }

    #[test]
    fn test_example() {
        let or = Or::new(vec![
            Box::new(Term::new([1, 2, 3].to_vec())),
            Box::new(Term::new([1, 7, 9].to_vec())),
        ]);

        let mut and = And::new(vec![
            Box::new(Term::new([1, 2, 7].to_vec())),
            Box::new(Term::new([1, 2, 4, 5, 7, 9].to_vec())),
            Box::new(or),
        ]);

        while and.next() != NO_MORE {
            println!("doc: {}, score: {}", and.doc_id(), and.score());
        }
    }
}
