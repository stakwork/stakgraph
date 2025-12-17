#[cfg(test)]
pub mod nextjs;

#[cfg(test)]
pub mod rust;

#[cfg(all(test, feature = "fulltest"))]
pub mod test_nodes;
