use crate::{Lang, lang::{Function, graphs::{Calls, Edge, EdgeType, Neo4jGraph, NodeData, NodeRef, NodeType, TransactionManager, queries::*}}};
use shared::Result;

impl Neo4jGraph{
        pub async fn add_instances_async(&self, nodes: Vec<NodeData>) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        for inst in &nodes {
            if let Some(of) = &inst.data_type {
                let class_nodes = self.find_nodes_by_name_async(NodeType::Class, of).await;
                if let Some(_class_node) = class_nodes.first() {
                    let queries = add_node_with_parent_query(
                        &NodeType::Instance,
                        inst,
                        &NodeType::File,
                        &inst.file,
                    );
                    for query in queries {
                        txn_manager.add_query(query);
                    }
                    let of_query = add_instance_of_query(inst, of);
                    txn_manager.add_query(of_query);
                }
            }
        }

        txn_manager.execute().await
    }
    pub async fn add_functions_async(&self, functions: Vec<Function>) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        for (function_node, method_of, reqs, dms, trait_operand, return_types, nested_in) in &functions {
            let queries = add_functions_query(
                function_node,
                method_of.as_ref(),
                reqs,
                dms,
                trait_operand.as_ref(),
                return_types,
                nested_in,
            );
            for query in queries {
                txn_manager.add_query(query);
            }
        }

        txn_manager.execute().await
    }
    pub async fn add_page_async(&self, page: (NodeData, Option<Edge>)) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let queries = add_page_query(&page.0, &page.1);

        let mut txn_manager = TransactionManager::new(&connection);
        for query in queries {
            txn_manager.add_query(query);
        }

        txn_manager.execute().await
    }

    pub async fn add_pages_async(&self, pages: Vec<(NodeData, Vec<Edge>)>) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let queries = add_pages_query(&pages);

        let mut txn_manager = TransactionManager::new(&connection);
        for query in queries {
            txn_manager.add_query(query);
        }
        txn_manager.execute().await
    }
    pub async fn add_endpoints_async(
        &self,
        endpoints: Vec<(NodeData, Option<Edge>)>,
    ) -> Result<()> {
        use std::collections::HashSet;
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        let mut to_add = Vec::new();
        let mut seen = HashSet::new();

        for (endpoint_data, handler_edge) in &endpoints {
            if endpoint_data.meta.get("handler").is_some() {
                let default_verb = "".to_string();
                let verb = endpoint_data.meta.get("verb").unwrap_or(&default_verb);
                let key = (
                    endpoint_data.name.clone(),
                    endpoint_data.file.clone(),
                    verb.clone(),
                );
                if seen.contains(&key) {
                    continue;
                }

                let exists = self
                    .find_endpoint_async(&endpoint_data.name, &endpoint_data.file, verb)
                    .await
                    .is_some();
                if !exists {
                    to_add.push((endpoint_data.clone(), handler_edge.clone()));
                    seen.insert(key);
                }
            }
        }

        for (endpoint_data, handler_edge) in &to_add {
            txn_manager.add_node(&NodeType::Endpoint, endpoint_data);
            if let Some(edge) = handler_edge {
                txn_manager.add_edge(edge);
            }
        }

        txn_manager.execute().await
    }

    pub async fn add_calls_async(
        &self,
        calls: (
            Vec<(Calls, Option<NodeData>, Option<NodeData>)>,
            Vec<(Calls, Option<NodeData>, Option<NodeData>)>,
            Vec<Edge>,
            Vec<Edge>,
        ),
        lang: &Lang,
    ) -> Result<()> {
        let (funcs, tests, int_tests, extras) = calls;
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        for (calls, ext_func, class_call) in &funcs {
            if let Some(cls_call) = class_call {
                txn_manager.add_node(&NodeType::Class, cls_call);
                let edge = Edge::new(
                    EdgeType::Calls,
                    NodeRef::from(calls.source.clone(), NodeType::Function),
                    NodeRef::from(cls_call.into(), NodeType::Class),
                );
                txn_manager.add_edge(&edge);
            }
            if calls.target.is_empty() {
                continue;
            }
            if let Some(ext_nd) = ext_func {
                txn_manager.add_node(&NodeType::Function, ext_nd);
                let edge = Edge::uses(calls.source.clone(), ext_nd);
                txn_manager.add_edge(&edge);
            } else {
                let edge: Edge = calls.clone().into();
                txn_manager.add_edge(&edge);
            }
        }
        for (test_call, ext_func, class_call) in &tests {
            let target_empty = test_call.target.is_empty();
            let has_class = class_call.is_some();

            if let Some(class_nd) = class_call {
                txn_manager.add_node(&NodeType::Class, class_nd);
                let edge = Edge::from_test_class_call(test_call, class_nd, lang, self);

                txn_manager.add_edge(&edge);
            }

            if let Some(ext_nd) = ext_func {
                txn_manager.add_node(&NodeType::Function, ext_nd);
                let edge = Edge::uses(test_call.source.clone(), ext_nd);
                txn_manager.add_edge(&edge);
            } else if target_empty && !has_class {
                continue;
            } else if !target_empty {
                let edge = Edge::from_test_call(test_call, lang, self);
                txn_manager.add_edge(&edge);
            }
        }
        for edge in int_tests {
            txn_manager.add_edge(&edge);
        }
        for edge in extras {
            txn_manager.add_edge(&edge);
        }

        txn_manager.execute().await
    }
}