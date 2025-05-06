use anyhow::Result;
use ast::lang::Lang;
use std::env;
use std::str::FromStr;

fn main() -> Result<()> {
    println!("Hello, world!");
    let args: Vec<String> = env::args().collect();
    let language = &args.get(1).expect("missing language argument");
    let lang = Lang::from_str(language)?;

    println!("=======> Class query <=======");
    println!("{}\n", lang.lang().class_definition_query());

    if let Some(instance_query) = lang.lang().instance_definition_query() {
        println!("=======> Instance query <=======");
        println!("{}\n", instance_query);
    }

    println!("=======> Function query <=======");
    println!("{}\n", lang.lang().function_definition_query());

    if let Some(tq) = lang.lang().test_query() {
        println!("=======> Test query <=======");
        println!("{}\n", tq);
    }

    println!("=======> Function call query <=======");
    println!("{}\n", lang.lang().function_call_query());

    for (i, endpoint_query) in lang.lang().endpoint_finders().iter().enumerate() {
        println!("=======> {i}. Endpoint query <=======");
        println!("{}\n", endpoint_query);
    }

    if let Some(request_query) = lang.lang().request_finder() {
        println!("=======> Request query <=======");
        println!("{}\n", request_query);
    }

    if let Some(dm_query) = lang.lang().data_model_query() {
        println!("=======> Data model query <=======");
        println!("{}\n", dm_query);
    }

    if let Some(dm_finder) = lang.lang().data_model_within_query() {
        println!("=======> Data model finder <=======");
        println!("{}\n", dm_finder);
    }

    println!("=======> Identifier query <=======");
    println!("{}\n", lang.lang().identifier_query());

    if let Some(itq) = lang.lang().integration_test_query() {
        println!("=======> Integration test query <=======");
        println!("{}\n", itq);
    }

    println!("=======> Library query <=======");
    if let Some(lq) = lang.lang().lib_query() {
        println!("{}\n", lq);
    }

    println!("=======> Imports query <=======");
    if let Some(iq) = lang.lang().imports_query() {
        println!("{}\n", iq);
    }

    println!("=======> Page query <=======");
    if let Some(rq) = lang.lang().page_query() {
        println!("{}\n", rq);
    }

    println!("=======> Variable query <=======");
    if let Some(vq) = lang.lang().variables_query() {
        println!("{}\n", vq);
    }

    Ok(())
}
