<?php
// @ast node: Endpoint "/" [verb=GET]
// @ast edge: Handler -> Function "get_handler_L45" "web.php"
// @ast node: Endpoint "/dashboard" [verb=GET]
// @ast edge: Handler -> Function "get_dashboard_handler_L49" "web.php"
// @ast node: Endpoint "/posts" [verb=GET]
// @ast node: Endpoint "/posts" [verb=POST]
// @ast node: Endpoint "/posts/create" [verb=GET]
// @ast node: Endpoint "/posts/{post}" [verb=DELETE]
// @ast node: Endpoint "/posts/{post}" [verb=GET]
// @ast node: Endpoint "/posts/{post}" [verb=PUT]
// @ast node: Endpoint "/posts/{post}/edit" [verb=GET]
// @ast node: Endpoint "/profile" [verb=DELETE]
// @ast node: Endpoint "/profile" [verb=GET]
// @ast node: Endpoint "/profile" [verb=PATCH]
// @ast node: Endpoint "/users" [verb=GET]
// @ast edge: Handler -> Function "index" "UserController.php"
// @ast node: Endpoint "/users" [verb=POST]
// @ast edge: Handler -> Function "store" "UserController.php"
// @ast node: Endpoint "/users/create" [verb=GET]
// @ast node: Endpoint "/users/{user}" [verb=DELETE]
// @ast node: Endpoint "/users/{user}" [verb=GET]
// @ast edge: Handler -> Function "show" "UserController.php"
// @ast node: Endpoint "/users/{user}" [verb=PUT]
// @ast node: Endpoint "/users/{user}/edit" [verb=GET]
// @ast node: Function "get_dashboard_handler_L49"
// @ast node: Function "get_handler_L45"
// @ast node: Import "import-imports-srctestingphprouteswebphp-29"

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\UserController;
use App\Http\Controllers\ProfileController;
use App\Http\Controllers\PostController;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "web" middleware group. Make something great!
|
*/

Route::get('/', function () {
    return view('welcome');
});

Route::get('/dashboard', function () {
    return view('dashboard');
})->middleware(['auth', 'verified'])->name('dashboard');

Route::middleware('auth')->group(function () {
    Route::get('/profile', [ProfileController::class, 'edit'])->name('profile.edit');
    Route::patch('/profile', [ProfileController::class, 'update'])->name('profile.update');
    Route::delete('/profile', [ProfileController::class, 'destroy'])->name('profile.destroy');
});

Route::resource('users', UserController::class);

Route::prefix('admin')->name('admin.')->group(function () {
    Route::resource('posts', PostController::class);
});

require __DIR__.'/auth.php';
