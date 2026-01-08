<?php

namespace App\Services;

use App\Models\User;

interface UserRepository
{
    public function create(array $data): User;
    public function find(int $id): ?User;
}
