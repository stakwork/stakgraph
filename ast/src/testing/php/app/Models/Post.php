<?php
// @ast node: Class "Post"
// @ast edge: Operand -> Function "user" "Post.php"
// @ast node: Function "user"
// @ast node: Import "import-imports-srctestingphpappmodelspostphp-8"

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class Post extends Model
{
    use HasFactory;

    protected $fillable = ['title', 'content', 'user_id'];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }
}
