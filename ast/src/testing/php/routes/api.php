<?php
// @ast node: Endpoint "/login" [verb=POST]
// @ast node: Endpoint "/posts" [verb=GET]
// @ast node: Endpoint "/posts" [verb=POST]
// @ast node: Endpoint "/posts/{post}" [verb=DELETE]
// @ast node: Endpoint "/posts/{post}" [verb=GET]
// @ast node: Endpoint "/posts/{post}" [verb=PUT]
// @ast node: Endpoint "/posts/{post}/comments" [verb=GET]
// @ast node: Endpoint "/posts/{post}/like" [verb=POST]
// @ast node: Endpoint "/register" [verb=POST]
// @ast node: Endpoint "/user" [verb=GET]
// @ast edge: Handler -> Function "get_user_handler_L40" "api.php"
// @ast node: Endpoint "/users" [verb=GET]
// @ast edge: Handler -> Function "index" "UserController.php"
// @ast node: Endpoint "/users" [verb=POST]
// @ast edge: Handler -> Function "store" "UserController.php"
// @ast node: Endpoint "/users/{user}" [verb=DELETE]
// @ast node: Endpoint "/users/{user}" [verb=GET]
// @ast edge: Handler -> Function "show" "UserController.php"
// @ast node: Endpoint "/users/{user}" [verb=PUT]
// @ast node: Function "get_user_handler_L40"
// @ast node: Import "import-imports-srctestingphproutesapiphp-23"

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Api\V1\UserController;
use App\Http\Controllers\Api\V1\PostController;
use App\Http\Controllers\AuthController;

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
|
| Here is where you can register API routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "api" middleware group. Make something great!
|
*/

Route::middleware('auth:sanctum')->get('/user', function (Request $request) {
    return $request->user();
});

Route::post('/login', [AuthController::class, 'login']);
Route::post('/register', [AuthController::class, 'register']);

Route::middleware(['auth:sanctum'])->group(function () {
    Route::apiResource('users', UserController::class);
    Route::apiResource('posts', PostController::class);
    
    Route::get('/posts/{post}/comments', [PostController::class, 'comments']);
    Route::post('/posts/{post}/like', [PostController::class, 'like']);
});
