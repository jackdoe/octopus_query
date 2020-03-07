# octopus - query arrays of integers

octopus is a query engine for arrays of integers (scoring only idf at the moment),
supports AND/OR/DisMax/Constant queries,

Example:

```
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

```