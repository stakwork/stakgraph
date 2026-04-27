import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AppComponent } from './app/app.component';
// @ast node: Import "import-imports-srctestingangularsrcmaints-0"

bootstrapApplication(AppComponent, appConfig)
  .catch((err) => console.error(err));
