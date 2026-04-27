<?php
// @ast node: Endpoint "/users/group_index" [verb=GET]
// @ast node: Endpoint "/users/group_store" [verb=POST]
// @ast edge: Handler -> Function "store" "UserController.php"
// @ast node: Import "import-imports-srctestingphprouteswebgroupsphp-6"

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\UserController;

Route::controller(UserController::class)->group(function () {
    Route::get('/users/group_index', 'index');
    Route::post('/users/group_store', 'store');
});
