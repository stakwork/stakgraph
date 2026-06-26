<?php
// @ast node: Class "UserServiceTest"
// @ast node: UnitTest "it_calculates_total_correctly"
// @ast node: UnitTest "test_it_can_instantiate_service"
// @ast node: UnitTest "testBasicAssertions"
// @ast node: UnitTest "test_register_user"
// @ast edge: Calls -> Function "registerUser" "UserService.php"
// @ast node: UnitTest "test_get_user"
// @ast edge: Calls -> Function "getUser" "UserService.php"
// @ast node: Var "$service"
// @ast node: Var "$service"
// @ast node: Var "$result"
// @ast node: Var "$service"
// @ast node: Var "$user"
// @ast node: Import "import-imports-srctestingphptestsunituserservicetestphp-18"

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

    public function test_register_user(): void
    {
        $service = new UserService();
        $result = $service->registerUser('test@example.com', 'password');
        $this->assertNotNull($result);
    }

    public function test_get_user(): void
    {
        $service = new UserService();
        $user = $service->getUser(1);
        $this->assertNotNull($user);
    }
}
