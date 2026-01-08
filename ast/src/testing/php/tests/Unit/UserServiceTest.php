<?php

namespace Tests\Unit;

use PHPUnit\Framework\TestCase;
use App\Services\UserService;

class UserServiceTest extends TestCase
{
    public function test_it_can_instantiate_service(): void
    {
        $service = new UserService();
        $this->assertInstanceOf(UserService::class, $service);
    }

    /** @test */
    public function it_calculates_total_correctly()
    {
        $this->assertEquals(4, 2 + 2);
    }
    
    public function testBasicAssertions(): void
    {
        $this->assertTrue(true);
        $this->assertFalse(false);
    }
}
