<?php
// @ast node: Endpoint "/posts-closure/{id}/like" [verb=POST]
// @ast edge: Handler -> Function "post_posts_closure_{id}_like_handler_L21" "anonymous_functions.php"
// @ast node: Endpoint "/status-arrow" [verb=GET]
// @ast edge: Handler -> Function "get_status_arrow_handler_L26" "anonymous_functions.php"
// @ast node: Endpoint "/users-closure" [verb=GET]
// @ast edge: Handler -> Function "get_users_closure_handler_L17" "anonymous_functions.php"
// @ast node: Function "get_status_arrow_handler_L26"
// @ast node: Function "get_users_closure_handler_L17"
// @ast node: Function "post_posts_closure_{id}_like_handler_L21"
// @ast node: Import "import-imports-srctestingphpanonymousfunctionsphp-12"

use Illuminate\Support\Facades\Route;
use App\Models\User;
use App\Models\Post;

// Laravel closure routes
Route::get('/users-closure', function() {
    return User::all();
});

Route::post('/posts-closure/{id}/like', function($id) {
    return Post::find($id)->like();
});

// Arrow function routes
Route::get('/status-arrow', fn() => ['status' => 'ok']);
