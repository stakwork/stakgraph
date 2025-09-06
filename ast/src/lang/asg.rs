use crate::lang::NodeType;
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use shared::error::{Error, Result};
use std::collections::BTreeMap;
use std::str::FromStr;
use crate::lang::Edge;

#[cfg(feature = "neo4j")]
use crate::lang::graphs::neo4j_utils::{boltmap_insert_int, boltmap_insert_str};
#[cfg(feature = "neo4j")]
use neo4rs::BoltMap;
#[cfg(feature = "neo4j")]
use neo4rs::Node as BoltNode;

pub struct UniqueKey {
    pub kind: NodeType,
    pub name: String,
    pub file: String,
    pub parent: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, Eq, PartialEq, PartialOrd, Ord)]
pub struct NodeKeys {
    pub name: String,
    pub file: String,
    pub start: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verb: Option<String>, // multiple endpoints can be defined on the same line, like ruby
}

impl NodeKeys {
    pub fn new(name: &str, file: &str, start: usize) -> Self {
        Self {
            name: name.to_string(),
            file: file.to_string(),
            start: start,
            verb: None,
        }
    }
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}
impl From<&NodeData> for NodeKeys {
    fn from(d: &NodeData) -> Self {
        NodeKeys {
            name: d.name.clone(),
            file: d.file.clone(),
            start: d.start,
            verb: d.meta.get("verb").map(|s| s.to_string()),
        }
    }
}
impl From<NodeData> for NodeKeys {
    fn from(d: NodeData) -> Self {
        NodeKeys {
            name: d.name,
            file: d.file,
            start: d.start,
            verb: d.meta.get("verb").map(|s| s.to_string()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Default, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct NodeData {
    pub name: String,
    pub file: String,
    pub body: String,
    pub start: usize,
    pub end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docs: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_type: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub meta: BTreeMap<String, String>,
}

impl Serialize for NodeData {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut named_fields_len = 5;
        if let Some(_) = &self.data_type {
            named_fields_len += 1;
        }
        if let Some(_) = &self.docs {
            named_fields_len += 1;
        }
        if let Some(_) = &self.hash {
            named_fields_len += 1;
        }
        let mut map = serializer.serialize_map(Some(self.meta.len() + named_fields_len))?;
        map.serialize_entry("name", &self.name)?;
        map.serialize_entry("file", &self.file)?;
        map.serialize_entry("body", &self.body)?;
        map.serialize_entry("start", &self.start)?;
        map.serialize_entry("end", &self.end)?;
        if let Some(data_type) = &self.data_type {
            map.serialize_entry("data_type", data_type)?;
        }
        if let Some(docs) = &self.docs {
            map.serialize_entry("docs", docs)?;
        }
        if let Some(hash) = &self.hash {
            map.serialize_entry("hash", hash)?;
        }
        // let mut map = serializer.serialize_map(Some(self.meta.len()))?;
        for (k, v) in self.meta.clone() {
            map.serialize_entry(&k, &v)?;
        }
        map.end()
    }
}

impl NodeData {
    pub fn name_file(name: &str, file: &str) -> Self {
        Self {
            name: name.to_string(),
            file: file.to_string(),
            ..Default::default()
        }
    }
    pub fn name_file_start(name: &str, file: &str, start: usize) -> Self {
        Self {
            name: name.to_string(),
            file: file.to_string(),
            start,
            ..Default::default()
        }
    }
    pub fn in_file(file: &str) -> Self {
        Self {
            file: file.to_string(),
            ..Default::default()
        }
    }
    pub fn add_verb(&mut self, verb: &str) {
        self.meta
            .insert("verb".to_string(), verb.to_string().to_uppercase());
    }
    pub fn add_module(&mut self, module: &str) {
        self.meta.insert("module".to_string(), module.to_string());
    }
    pub fn add_operand(&mut self, operand: &str) {
        self.meta.insert("operand".to_string(), operand.to_string());
    }
    pub fn add_source_link(&mut self, url: &str) {
        self.meta.insert("source_link".to_string(), url.to_string());
    }
    pub fn add_handler(&mut self, handler: &str) {
        self.meta.insert("handler".to_string(), handler.to_string());
    }
    pub fn add_version(&mut self, version: &str) {
        self.meta.insert("version".to_string(), version.to_string());
    }
    pub fn add_action(&mut self, action: &str) {
        self.meta.insert("action".to_string(), action.to_string());
    }
    pub fn add_group(&mut self, verb: &str) {
        self.meta.insert("group".to_string(), verb.to_string());
    }
    pub fn add_function_type(&mut self, function_type: &str) {
        self.meta
            .insert("function_type".to_string(), function_type.to_string());
    }
    pub fn add_parent(&mut self, parent: &str) {
        self.meta.insert("parent".to_string(), parent.to_string());
    }
    pub fn add_includes(&mut self, modules: &str) {
        self.meta
            .insert("includes".to_string(), modules.to_string());
    }
    pub fn add_implements(&mut self, trait_name: &str) {
        self.meta
            .insert("implements".to_string(), trait_name.to_string());
    }
    pub fn add_component(&mut self) {
        self.meta.insert("component".to_string(), "true".to_string());
    }
    pub fn add_test_kind(&mut self, test_kind: &str) {
        self.meta.insert("test_kind".to_string(), test_kind.to_string());
    }
    pub fn test_covered(&mut self, covered: bool) {
        if covered {
            self.meta.insert("test_covered".to_string(), "true".into());
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Operand {
    pub source: NodeKeys,
    pub target: NodeKeys,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Calls {
    pub source: NodeKeys,
    pub target: NodeKeys,
    pub operand: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestRecord {
    pub node: NodeData,
    pub kind: NodeType,
    pub edges: Vec<Edge>,
}

impl TestRecord {
    pub fn new(node: NodeData, kind: NodeType, edge: Option<Edge>) -> Self {
        let mut edges = Vec::new();
        if let Some(e) = edge { edges.push(e); }
        Self { node, kind, edges }
    }
    pub fn test_kind(&self) -> String {
        self.node
            .meta
            .get("test_kind")
            .cloned()
            .unwrap_or_else(|| match self.kind {
                NodeType::IntegrationTest => "integration".into(),
                NodeType::E2eTest => "e2e".into(),
                _ => "unit".into(),
            })
    }
}

impl FromStr for NodeType {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "Class" => Ok(NodeType::Class),
            "Trait" => Ok(NodeType::Trait),
            "Instance" => Ok(NodeType::Instance),
            "Function" => Ok(NodeType::Function),
            "UnitTest" => Ok(NodeType::UnitTest),
            "IntegrationTest" => Ok(NodeType::IntegrationTest),
            "E2etest" => Ok(NodeType::E2eTest),
            "File" => Ok(NodeType::File),
            "Repository" => Ok(NodeType::Repository),
            "Endpoint" => Ok(NodeType::Endpoint),
            "Request" => Ok(NodeType::Request),
            "Datamodel" => Ok(NodeType::DataModel),
            "Feature" => Ok(NodeType::Feature),
            "Page" => Ok(NodeType::Page),
            "Var" => Ok(NodeType::Var),
            _ => Err(Error::Custom(format!("Invalid NodeType string: {}", s))),
        }
    }
}
impl ToString for NodeType {
    fn to_string(&self) -> String {
        match self {
            NodeType::Repository => "Repository".to_string(),
            NodeType::Directory => "Directory".to_string(),
            NodeType::File => "File".to_string(),
            NodeType::Language => "Language".to_string(),
            NodeType::Library => "Library".to_string(),
            NodeType::Class => "Class".to_string(),
            NodeType::Trait => "Trait".to_string(),
            NodeType::Import => "Import".to_string(),
            NodeType::Instance => "Instance".to_string(),
            NodeType::Function => "Function".to_string(),
            NodeType::UnitTest => "UnitTest".to_string(),
            NodeType::IntegrationTest => "IntegrationTest".to_string(),
            NodeType::E2eTest => "E2etest".to_string(),
            NodeType::Endpoint => "Endpoint".to_string(),
            NodeType::Request => "Request".to_string(),
            NodeType::DataModel => "Datamodel".to_string(),
            NodeType::Feature => "Feature".to_string(),
            NodeType::Page => "Page".to_string(),
            NodeType::Var => "Var".to_string(),
        }
    }
}

const SEP: &str = "|:|";

impl FromStr for UniqueKey {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        let arr = s.split(SEP).collect::<Vec<&str>>();
        if arr.len() != 3 && arr.len() != 4 {
            return Err(Error::Custom(format!("Invalid UniqueKey string: {}", s)));
        }
        let kind = NodeType::from_str(arr[0])?;
        Ok(UniqueKey {
            kind,
            name: arr[1].to_string(),
            file: arr[2].to_string(),
            parent: arr.get(3).map(|s| s.to_string()),
        })
    }
}
impl ToString for UniqueKey {
    fn to_string(&self) -> String {
        let mut s = format!(
            "{}{}{}{}{}",
            self.kind.to_string(),
            SEP,
            self.name,
            SEP,
            self.file
        );
        if let Some(parent) = &self.parent {
            s.push_str(&format!("{SEP}{parent}"));
        }
        s
    }
}

impl ToString for Operand {
    fn to_string(&self) -> String {
        let s = format!("{:?}", self.source.name);
        s //Given that the source is a class
    }
}
#[cfg(feature = "neo4j")]
impl From<&NodeData> for BoltMap {
    fn from(node_data: &NodeData) -> Self {
        let mut map = BoltMap::new();
        boltmap_insert_str(&mut map, "name", &node_data.name);
        boltmap_insert_str(&mut map, "file", &node_data.file);
        boltmap_insert_str(&mut map, "body", &node_data.body);
        boltmap_insert_int(&mut map, "start", node_data.start as i64);
        boltmap_insert_int(&mut map, "end", node_data.end as i64);
        if let Some(ref docs) = node_data.docs {
            boltmap_insert_str(&mut map, "docs", docs);
        }
        if let Some(ref hash) = node_data.hash {
            boltmap_insert_str(&mut map, "hash", hash);
        }
        if let Some(ref data_type) = node_data.data_type {
            boltmap_insert_str(&mut map, "data_type", data_type);
        }
        for (k, v) in &node_data.meta {
            boltmap_insert_str(&mut map, k, v);
        }

        map
    }
}

#[cfg(feature = "neo4j")]
impl TryFrom<&BoltNode> for NodeData {
    type Error = Error;
    fn try_from(node: &BoltNode) -> Result<Self> {
        let mut meta = BTreeMap::new();
        let known_fields = [
            "name",
            "file",
            "body",
            "start",
            "end",
            "docs",
            "hash",
            "data_type",
        ];
        for k in node.keys() {
            if !known_fields.contains(&k) {
                if let Ok(val) = node.get::<String>(k) {
                    meta.insert(k.to_string(), val);
                }
            }
        }
        Ok(NodeData {
            name: node.get::<String>("name").unwrap_or_default(),
            file: node.get::<String>("file").unwrap_or_default(),
            body: node.get::<String>("body").unwrap_or_default(),
            start: node.get::<i64>("start").unwrap_or(0) as usize,
            end: node.get::<i64>("end").unwrap_or(0) as usize,
            docs: node.get::<String>("docs").ok(),
            hash: node.get::<String>("hash").ok(),
            data_type: node.get::<String>("data_type").ok(),
            meta,
        })
    }
}
