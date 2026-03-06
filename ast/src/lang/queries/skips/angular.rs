const ANGULAR_LIFECYCLE: [&str; 8] = [
    "ngOnInit",
    "ngOnDestroy",
    "ngAfterViewInit",
    "ngAfterViewChecked",
    "ngAfterContentInit",
    "ngAfterContentChecked",
    "ngDoCheck",
    "ngOnChanges",
];

const RXJS_METHODS: [&str; 15] = [
    "subscribe",
    "next",
    "complete",
    "error",
    "pipe",
    "map",
    "filter",
    "tap",
    "takeUntil",
    "switchMap",
    "mergeMap",
    "flatMap",
    "debounceTime",
    "distinctUntilChanged",
    "unsubscribe",
];

const ANGULAR_CORE: [&str; 12] = [
    "Injectable",
    "Component",
    "Directive",
    "Pipe",
    "NgModule",
    "Input",
    "Output",
    "ViewChild",
    "ContentChild",
    "HostListener",
    "HostBinding",
    "OnInit",
];

const DECORATORS: [&str; 8] = [
    "Logger",
    "Memoize",
    "Debounce",
    "Throttle",
    "Validate",
    "Cache",
    "Retry",
    "Deprecated",
];

pub fn should_skip(called: &str, operand: &Option<String>) -> bool {
    if ANGULAR_LIFECYCLE.contains(&called)
        || RXJS_METHODS.contains(&called)
        || ANGULAR_CORE.contains(&called)
        || DECORATORS.contains(&called)
    {
        return true;
    }

    if let Some(op) = operand {
        if matches!(
            op.as_str(),
            "BehaviorSubject"
                | "Subject"
                | "Observable"
                | "ReplaySubject"
                | "HttpClient"
                | "ActivatedRoute"
                | "Router"
                | "FormBuilder"
                | "FormControl"
                | "FormGroup"
        ) {
            return true;
        }
    }

    false
}
