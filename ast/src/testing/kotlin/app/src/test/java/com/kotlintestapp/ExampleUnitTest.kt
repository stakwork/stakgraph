package com.kotlintestapp
// @ast node: Class "ExampleUnitTest"
// @ast node: UnitTest "addition_isCorrect"
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
}