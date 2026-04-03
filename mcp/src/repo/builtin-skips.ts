const TS_ARRAY_METHODS = [
  "push","pop","shift","unshift","slice","splice","concat","join","reverse","sort",
  "indexOf","lastIndexOf","forEach","map","filter","reduce","reduceRight","every",
  "some","find","findIndex","includes","flat","flatMap","fill","copyWithin","entries",
  "keys","values","at",
];

const TS_STRING_METHODS = [
  "charAt","charCodeAt","concat","includes","indexOf","lastIndexOf","match","matchAll",
  "replace","replaceAll","search","slice","split","substring","toLowerCase","toUpperCase",
  "trim","trimStart","trimEnd","padStart","padEnd","repeat","startsWith","endsWith",
  "localeCompare","normalize","at","codePointAt","fromCharCode","fromCodePoint",
];

const TS_OBJECT_METHODS = [
  "hasOwnProperty","isPrototypeOf","propertyIsEnumerable","toLocaleString","toString",
  "valueOf","keys","values","entries","assign","create","defineProperty","freeze","seal",
  "preventExtensions",
];

const TS_ASYNC_METHODS = ["then","catch","finally","all","race","allSettled"];

const TS_DOM_METHODS = [
  "addEventListener","removeEventListener","querySelector","querySelectorAll",
  "getElementById","getElementsByClassName","getElementsByTagName","appendChild",
  "removeChild","replaceChild","insertBefore","cloneNode","setAttribute","getAttribute",
  "removeAttribute","classList","focus","blur","click","submit",
];

const TS_TEST_FRAMEWORK = [
  "describe","it","test","expect","beforeEach","afterEach","beforeAll","afterAll",
  "vi","jest","suite","specify","assert",
];

const TS_JSX_HTML_ELEMENTS = [
  "div","span","p","section","article","nav","header","footer","main","aside","address",
  "h1","h2","h3","h4","h5","h6","form","input","button","label","select","textarea",
  "fieldset","legend","datalist","option","optgroup","ul","ol","li","dl","dt","dd",
  "table","tbody","thead","tfoot","tr","td","th","caption","colgroup","col","img",
  "video","audio","source","track","canvas","iframe","embed","object","param","map",
  "area","strong","em","code","pre","kbd","var","samp","mark","ins","del","sub","sup",
  "small","b","i","u","s","q","bdi","bdo","details","summary","dialog","meta","link",
  "style","svg","g","path","rect","circle","ellipse","line","polyline","polygon","text",
  "tspan","defs","use","symbol","marker","clipPath","mask","pattern","image",
  "foreignObject","switch","a","view","animate","animateMotion","animateTransform","set",
  "desc","title","metadata","stop","linearGradient","radialGradient","feBlend",
  "feColorMatrix","feComponentTransfer","feComposite","feConvolveMatrix",
  "feDiffuseLighting","feDisplacementMap","feFlood","feGaussianBlur","feImage","feMerge",
  "feMorphology","feOffset","feSpecularLighting","feTile","feTurbulence","feDistantLight",
  "fePointLight","feSpotLight",
];

const TS_BUILTIN_OBJECTS = [
  "console","Math","Number","BigInt","Array","Object","Map","Set","WeakMap","WeakSet",
  "String","JSON","Intl","RegExp","Promise","Date","ArrayBuffer","DataView","Uint8Array",
  "Int32Array","Float32Array","Float64Array","Reflect","Proxy","Symbol","Error","process",
  "Buffer","require","module","exports","global","window","document","navigator",
  "localStorage","sessionStorage","fetch","history","location","screen","performance",
];

const TS_SCHEMA_BUILDERS = [
  "string","number","boolean","object","array","enum","optional","nullable","union",
  "intersection","literal","tuple","record","any","unknown","never","void","null",
  "undefined","tool","describe",
];

const PYTHON_BUILTINS = [
  "len","print","range","enumerate","zip","map","filter","sorted","reversed","sum",
  "min","max","any","all","isinstance",
];

const PYTHON_LIST_METHODS = [
  "append","extend","pop","remove","sort","reverse","clear","copy","count","index",
  "insert","len",
];

const PYTHON_DICT_METHODS = [
  "keys","values","items","get","pop","update","clear","copy","setdefault","fromkeys",
  "popitem","len",
];

const PYTHON_STRING_METHODS = [
  "split","join","strip","lstrip","rstrip","replace","upper","lower","capitalize",
  "title","find","index","startswith","endswith","isdigit","isalpha","format","encode",
  "decode","len",
];

const GO_BUILTIN_FUNCTIONS = [
  "len","cap","make","append","copy","delete","panic","recover","close","new",
];

const GO_STDLIB_PACKAGES = [
  "fmt","log","errors","strings","strconv","time","math","os","io","bufio","bytes",
  "sync","json","http","net","context","regexp","sort","path","filepath",
];

const RUST_VEC = [
  "push","pop","len","is_empty","iter","iter_mut","map","filter","collect","into_iter",
  "get","first","last","contains","sort","reverse","retain","dedup","clear","extend",
];

const RUST_OPTION = [
  "unwrap","expect","map","and_then","or_else","is_some","is_none","unwrap_or",
  "unwrap_or_else","ok_or","ok_or_else","take","filter","flatten","as_ref",
];

const RUST_RESULT = [
  "unwrap","expect","map","and_then","or_else","is_ok","is_err","unwrap_or",
  "unwrap_or_else","map_err","ok","err","as_ref","unwrap_or_default","expect_err",
];

const RUST_STRING = [
  "len","is_empty","push_str","to_string","clone","as_str","into_bytes","chars","bytes",
  "split","trim","to_lowercase","to_uppercase","replace","contains",
];

const RUST_ITERATOR = [
  "map","filter","collect","fold","for_each","any","all","find","position","count",
  "take","skip","chain","zip","enumerate","flat_map","flatten","cloned","copied","sum",
];

const RUST_COMMON_TRAITS = [
  "clone","to_owned","into","from","as_ref","as_mut","default","to_string","fmt","drop",
];

const RUST_PATH = [
  "parent","file_name","extension","file_stem","to_path_buf","to_str","display",
  "exists","is_file","is_dir","is_absolute","is_relative","canonicalize","read_dir",
  "join","with_extension","with_file_name","starts_with","ends_with","strip_prefix",
  "components","ancestors","metadata","symlink_metadata","read_link",
];

const RUST_OSSTR = ["to_str","to_string_lossy","to_os_string","len","is_empty"];

const RUST_CONSTRUCTORS = [
  "new","default","Ok","Err","Some","None","Custom","with_capacity","from","into",
];

const RUST_WALKDIR = [
  "new","min_depth","max_depth","follow_links","max_open","sort_by","filter_entry",
  "into_iter","path","file_name",
];

const RUBY_TEST_FRAMEWORK = [
  "to","not_to","to_not","eq","eql","be","be_a","be_an","be_nil","be_truthy",
  "be_falsey","be_true","be_false","be_empty","be_blank","be_present","include",
  "match","raise_error","change","respond_to",
];

const RUBY_ARRAY = [
  "push","pop","shift","unshift","map","select","reject","each","compact","flatten",
  "uniq","sort","reverse","first","last","size","length","empty?","include?","join",
];

const RUBY_STRING = [
  "upcase","downcase","strip","split","gsub","sub","length","size","empty?","include?",
  "start_with?","end_with?","chars","bytes","to_s",
];

const RUBY_HASH = [
  "keys","values","merge","fetch","dig","each","map","select","reject","empty?","size",
  "length",
];

const RUBY_ENUMERABLE = [
  "find","inject","reduce","zip","any?","all?","none?","one?","min","max",
];

const RUBY_KEYWORDS = [
  "expect","describe","it","context","before","after","let","subject","have_",
];

const JAVA_STRING = [
  "length","isEmpty","substring","toLowerCase","toUpperCase","trim","split","replace",
  "replaceAll","concat","contains","startsWith","endsWith","indexOf","lastIndexOf",
  "charAt","matches","equals","equalsIgnoreCase","valueOf",
];

const JAVA_LIST = [
  "add","remove","get","size","isEmpty","contains","clear","addAll","removeAll",
  "indexOf","set","subList","toArray","iterator","stream",
];

const JAVA_COLLECTION = [
  "size","isEmpty","contains","iterator","toArray","add","remove","clear","stream",
  "forEach",
];

const JAVA_MAP = [
  "put","putAll","putIfAbsent","get","getOrDefault","containsKey","containsValue",
  "remove","keySet","values","entrySet",
];

const JAVA_OPTIONAL = [
  "isPresent","isEmpty","get","orElse","orElseGet","orElseThrow","map","ifPresent",
];

const JAVA_COMMON_CLASSES = [
  "System","Math","String","Integer","Double","Long","Boolean","Collections","Arrays",
  "Objects","Optional","Stream","List","Map","Set","Collectors",
];

const C_STDLIB = [
  "printf","fprintf","sprintf","snprintf","scanf","fscanf","sscanf","fgets","fputs",
  "fopen","fclose","fread","fwrite","fseek","ftell","rewind","fflush","putchar",
  "getchar","puts","gets","malloc","free","calloc","realloc","exit","abort","atexit",
  "atoi","atof","atol","strtol","strtod","rand","srand","qsort","bsearch","abs","labs",
  "strcpy","strncpy","strcat","strncat","strcmp","strncmp","strlen","strchr","strrchr",
  "strstr","memcpy","memmove","memset","memcmp","memchr","sqrt","pow","sin","cos",
  "tan","floor","ceil",
];

const C_POSIX = [
  "pthread_create","pthread_join","pthread_exit","pthread_detach","pthread_mutex_init",
  "pthread_mutex_lock","pthread_mutex_unlock","pthread_mutex_destroy",
  "pthread_cond_init","pthread_cond_wait","pthread_cond_signal",
  "pthread_cond_broadcast","time","clock","difftime","strftime","localtime","gmtime",
  "read","write","close","open","pipe","fork","execve","sleep","usleep","signal",
  "raise","kill",
];

const C_LIBRARIES = [
  "stdio","stdlib","string","math","time","pthread","unistd","signal","errno","assert",
  "ctype","limits","float","stddef","stdint",
];

const CPP_STDLIB = [
  "push_back","pop_back","insert","erase","find","clear","size","begin","end","get",
  "make_shared","make_unique","move","swap","emplace_back","reserve","resize","front",
  "back","at",
];

const CPP_SQLITE = [
  "sqlite3_open","sqlite3_close","sqlite3_prepare","sqlite3_prepare_v2","sqlite3_step",
  "sqlite3_reset","sqlite3_finalize","sqlite3_bind_int","sqlite3_bind_text",
  "sqlite3_bind_double","sqlite3_column_int","sqlite3_column_text",
  "sqlite3_column_double","sqlite3_errmsg","sqlite3_exec",
];

const CPP_CROW = [
  "CROW_ROUTE","CROW_WEBSOCKET_ROUTE","CROW_BP_ROUTE","methods","enable_cors","port",
  "multithreaded_service","validate_json",
];

const CPP_HTTP = [
  "status","set_header","body","json","response","request","query_string","headers",
  "path","method",
];

const PHP_BUILTINS = [
  "count","isset","empty","in_array","array_map","array_filter","array_reduce",
  "json_encode","json_decode","strlen","explode","implode","trim",
];

const SVELTE_LIFECYCLE = [
  "onMount","onDestroy","beforeUpdate","afterUpdate","tick",
  "createEventDispatcher","bind","animate",
];

const SVELTE_WEB_API = [
  "fetch","then","catch","finally","json","text","blob","ok","status","headers","body",
  "clone","JSON","parse","stringify","addEventListener","removeEventListener",
  "preventDefault","stopPropagation","querySelector",
];

const SVELTE_STORES = [
  "writable","readable","derived","subscribe","set","update","get","unsubscribe",
];

const SVELTE_ANIMATIONS = [
  "transition","animate","fade","fly","slide","scale","draw","crossfade","blur","bounce",
];

const ANGULAR_LIFECYCLE = [
  "ngOnInit","ngOnDestroy","ngAfterViewInit","ngAfterViewChecked","ngAfterContentInit",
  "ngAfterContentChecked","ngDoCheck","ngOnChanges",
];

const ANGULAR_RXJS = [
  "subscribe","next","complete","error","pipe","map","filter","tap","takeUntil",
  "switchMap","mergeMap","flatMap","debounceTime","distinctUntilChanged","unsubscribe",
];

const ANGULAR_CORE = [
  "Injectable","Component","Directive","Pipe","NgModule","Input","Output","ViewChild",
  "ContentChild","HostListener","HostBinding","OnInit",
];

const ANGULAR_DECORATORS = [
  "Logger","Memoize","Debounce","Throttle","Validate","Cache","Retry","Deprecated",
];

function toSet(...arrays: string[][]): Set<string> {
  const s = new Set<string>();
  for (const arr of arrays) {
    for (const v of arr) s.add(v);
  }
  return s;
}

const SKIP_SETS: Record<string, Set<string>> = {
  typescript: toSet(
    TS_ARRAY_METHODS, TS_STRING_METHODS, TS_OBJECT_METHODS, TS_ASYNC_METHODS,
    TS_DOM_METHODS, TS_TEST_FRAMEWORK, TS_JSX_HTML_ELEMENTS, TS_BUILTIN_OBJECTS,
    TS_SCHEMA_BUILDERS,
  ),
  javascript: toSet(
    TS_ARRAY_METHODS, TS_STRING_METHODS, TS_OBJECT_METHODS, TS_ASYNC_METHODS,
    TS_DOM_METHODS, TS_TEST_FRAMEWORK, TS_JSX_HTML_ELEMENTS, TS_BUILTIN_OBJECTS,
    TS_SCHEMA_BUILDERS,
  ),
  python: toSet(
    PYTHON_BUILTINS, PYTHON_LIST_METHODS, PYTHON_DICT_METHODS, PYTHON_STRING_METHODS,
  ),
  go: toSet(GO_BUILTIN_FUNCTIONS, GO_STDLIB_PACKAGES),
  rust: toSet(
    RUST_VEC, RUST_OPTION, RUST_RESULT, RUST_STRING, RUST_ITERATOR,
    RUST_COMMON_TRAITS, RUST_PATH, RUST_OSSTR, RUST_CONSTRUCTORS, RUST_WALKDIR,
  ),
  ruby: toSet(
    RUBY_TEST_FRAMEWORK, RUBY_ARRAY, RUBY_STRING, RUBY_HASH, RUBY_ENUMERABLE,
    RUBY_KEYWORDS,
  ),
  java: toSet(
    JAVA_STRING, JAVA_LIST, JAVA_COLLECTION, JAVA_MAP, JAVA_OPTIONAL,
    JAVA_COMMON_CLASSES,
  ),
  kotlin: toSet(
    JAVA_STRING, JAVA_LIST, JAVA_COLLECTION, JAVA_MAP, JAVA_OPTIONAL,
    JAVA_COMMON_CLASSES,
  ),
  swift: toSet(
    JAVA_STRING, JAVA_LIST, JAVA_COLLECTION, JAVA_MAP, JAVA_OPTIONAL,
    JAVA_COMMON_CLASSES,
  ),
  c: toSet(C_STDLIB, C_POSIX, C_LIBRARIES),
  cpp: toSet(C_STDLIB, C_POSIX, C_LIBRARIES, CPP_STDLIB, CPP_SQLITE, CPP_CROW, CPP_HTTP),
  csharp: toSet(
    JAVA_STRING, JAVA_LIST, JAVA_COLLECTION, JAVA_MAP, JAVA_OPTIONAL,
    JAVA_COMMON_CLASSES,
  ),
  php: toSet(PHP_BUILTINS),
  svelte: toSet(
    SVELTE_LIFECYCLE, SVELTE_WEB_API, SVELTE_STORES, SVELTE_ANIMATIONS,
    TS_ARRAY_METHODS, TS_STRING_METHODS, TS_OBJECT_METHODS, TS_ASYNC_METHODS,
    TS_DOM_METHODS, TS_TEST_FRAMEWORK, TS_JSX_HTML_ELEMENTS, TS_BUILTIN_OBJECTS,
    TS_SCHEMA_BUILDERS,
  ),
  angular: toSet(
    ANGULAR_LIFECYCLE, ANGULAR_RXJS, ANGULAR_CORE, ANGULAR_DECORATORS,
    TS_ARRAY_METHODS, TS_STRING_METHODS, TS_OBJECT_METHODS, TS_ASYNC_METHODS,
    TS_DOM_METHODS, TS_TEST_FRAMEWORK, TS_JSX_HTML_ELEMENTS, TS_BUILTIN_OBJECTS,
    TS_SCHEMA_BUILDERS,
  ),
};

const EXT_TO_LANGUAGE: Record<string, string> = {};

const LANGUAGE_EXTENSIONS: Record<string, string[]> = {
  javascript: [".js", ".jsx", ".cjs", ".mjs"],
  typescript: [".ts", ".tsx", ".d.ts"],
  python: [".py", ".pyi", ".pyw"],
  ruby: [".rb", ".rake", ".gemspec"],
  go: [".go"],
  rust: [".rs"],
  java: [".java"],
  kotlin: [".kt", ".kts"],
  swift: [".swift"],
  c: [".c", ".h"],
  cpp: [".cpp", ".cc", ".cxx", ".hpp", ".hxx"],
  csharp: [".cs"],
  php: [".php"],
  svelte: [".svelte"],
};

for (const [lang, exts] of Object.entries(LANGUAGE_EXTENSIONS)) {
  for (const ext of exts) {
    EXT_TO_LANGUAGE[ext] = lang;
  }
}

const ANGULAR_SUFFIXES = [
  ".component.ts", ".service.ts", ".directive.ts", ".pipe.ts", ".module.ts",
];

function languageFromFile(file: string): string | null {
  if (!file) return null;
  const lower = file.toLowerCase();

  for (const suffix of ANGULAR_SUFFIXES) {
    if (lower.endsWith(suffix)) return "angular";
  }

  const dotIdx = lower.lastIndexOf(".");
  if (dotIdx === -1) return null;
  const ext = lower.slice(dotIdx);
  return EXT_TO_LANGUAGE[ext] || null;
}

export function shouldSkipDescription(
  name: string,
  file: string,
  labels: string[],
): boolean {
  if (!labels.includes("Function")) return false;

  const lang = languageFromFile(file);
  if (!lang) return false;

  const skipSet = SKIP_SETS[lang];
  if (!skipSet) return false;

  return skipSet.has(name);
}
