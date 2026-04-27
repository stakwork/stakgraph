import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';
import { config } from './app/app.config.server';
// @ast node: Import "import-imports-srctestingangularsrcmainserverts-0"

const bootstrap = () => bootstrapApplication(AppComponent, config);

export default bootstrap;
