#![feature(test)]
extern crate test;

use test::Bencher;

#[bench]
fn bench_string_allocation(b: &mut Bencher) {
    b.iter(|| {
        let s = String::from("test");
        s.len()
    });
}

#[bench]
fn bench_vector_push(b: &mut Bencher) {
    b.iter(|| {
        let mut v = Vec::new();
        for i in 0..100 {
            v.push(i);
        }
        v
    });
}
