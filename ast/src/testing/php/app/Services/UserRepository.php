<?php
// @ast node: Class "UserRepository"
// @ast node: Function "create"
// @ast node: Function "find"
// @ast node: Import "import-imports-srctestingphpappservicesuserrepositoryphp-8"

namespace App\Services;

use App\Models\User;

interface UserRepository
{
    public function create(array $data): User;
    public function find(int $id): ?User;
}
