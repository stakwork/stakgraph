package com.kotlintestapp
// @ast node: Class "ExampleUnitTest"
// @ast node: UnitTest "addition_isCorrect"
// @ast node: UnitTest "test_fetchUsers"
// @ast edge: Calls -> Function "fetchUsers" "HomeViewModel.kt"
// @ast node: UnitTest "test_personList"
// @ast edge: Calls -> Function "PersonList" "MainActivity.kt"
// @ast edge: Calls -> Function "PersonItem" "MainActivity.kt"
// @ast node: Import "import-imports-srctestingkotlinappsrctestjavacomkotlintestappexampleunittestkt-0"

import org.junit.Test

import org.junit.Assert.*

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class ExampleUnitTest {
    @Test
    fun addition_isCorrect() {
        assertEquals(4, 2 + 2)
    }

    @Test
    fun test_fetchUsers() {
        val viewModel = HomeViewModel()
        viewModel.fetchUsers()
        assertNotNull(viewModel)
    }

    @Test
    fun test_personList() {
        PersonList(listOf())
        PersonItem(Person("Test", "test@example.com"))
        assertTrue(true)
    }
}