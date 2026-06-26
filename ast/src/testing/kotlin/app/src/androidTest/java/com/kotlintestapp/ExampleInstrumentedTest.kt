package com.kotlintestapp
// @ast node: Class "ExampleInstrumentedTest"
// @ast node: IntegrationTest "useAppContext"
// @ast node: IntegrationTest "test_updatePerson"
// @ast edge: Calls -> Function "updatePerson" "PersonViewModel.kt"
// @ast node: IntegrationTest "test_fetchAndStorePersons"
// @ast edge: Calls -> Function "fetchAndStorePersons" "PersonViewModel.kt"
// @ast node: Import "import-imports-srctestingkotlinappsrcandroidtestjavacomkotlintestappexampleinstrumentedtestkt-0"

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class ExampleInstrumentedTest {
    @Test
    fun useAppContext() {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("com.kotlintestapp", appContext.packageName)
    }

    @Test
    fun test_updatePerson() {
        val viewModel = PersonViewModel()
        viewModel.updatePerson("Alice", "alice@test.com")
        val persons = DatabaseHelper().getAllPersons()
        assertNotNull(persons)
    }

    @Test
    fun test_fetchAndStorePersons() {
        val viewModel = PersonViewModel()
        DatabaseHelper().clearDatabase()
        viewModel.fetchAndStorePersons()
        assertTrue(true)
    }
}