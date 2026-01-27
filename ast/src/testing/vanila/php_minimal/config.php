<?php

define('DB_HOST', getenv('DB_HOST') ?: 'localhost');
define('DB_NAME', getenv('DB_NAME') ?: 'test_db');
define('DB_USER', getenv('DB_USER') ?: 'root');
define('DB_PASS', getenv('DB_PASS') ?: '');

define('APP_ENV', getenv('APP_ENV') ?: 'development');
define('DEBUG', APP_ENV === 'development');
