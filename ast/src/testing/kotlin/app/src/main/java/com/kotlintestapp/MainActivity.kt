package com.kotlintestapp
// @ast node: Class "MainActivity"
// @ast edge: Operand -> Function "onCreate" "MainActivity.kt"
// @ast node: Function "onCreate"
// @ast edge: Calls -> Function "PersonList" "MainActivity.kt"
// @ast node: Function "PersonItem"
// @ast node: Function "PersonList"
// @ast edge: Calls -> Function "PersonItem" "MainActivity.kt"
// @ast node: Function "UpdateProfileDialog"
// @ast node: Import "import-imports-srctestingkotlinappsrcmainjavacomkotlintestappmainactivitykt-0"

import android.annotation.SuppressLint
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import com.kotlintestapp.db.Person
import com.kotlintestapp.ui.theme.KotlinTestAppTheme
import com.kotlintestapp.viewModels.PersonViewModel

class MainActivity : ComponentActivity() {

    private lateinit var personViewModel: PersonViewModel

    @SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        personViewModel = PersonViewModel(this)

        enableEdgeToEdge()

        setContent {
            KotlinTestAppTheme {
                Scaffold(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(top = WindowInsets.statusBars.asPaddingValues().calculateTopPadding())
                ) {
                    PersonList(viewModel = personViewModel)
                }
            }
        }
    }
}

@Composable
fun PersonList(viewModel: PersonViewModel) {
    // Observing the list of Persons
    val persons by remember { viewModel.allPersons }

    LazyColumn(modifier = Modifier.fillMaxSize()) {
        items(persons) { person ->
            PersonItem(person) {
                val updatedPerson = com.kotlintestapp.models.Person(
                    person.id.toInt(),
                    person.alias,
                    person.img ?: "",
                    person.publicKey,
                    person.routeHint ?: ""
                )

                viewModel.updatePerson(updatedPerson)
            }
        }
    }
}

@Composable
fun PersonItem(person: Person, onClick: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp)
            .pointerInput(Unit) {
                detectTapGestures(onTap = {
                    onClick()
                })
            },
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(text = "Alias: ${person.alias}")
            Text(text = "Public Key: ${person.publicKey}")
            Text(text = "Route Hint: ${person.routeHint ?: "N/A"}")
        }
    }
}

@Composable
fun UpdateProfileDialog(
    person: Person,
    onDismiss: () -> Unit,
    onConfirm: (Person) -> Unit
) {
    var updatedTitle by remember { mutableStateOf(TextFieldValue(person.alias)) }

    AlertDialog(
        onDismissRequest = { onDismiss() },
        title = { Text("Update Profile") },
        text = {
            Column {
                Text("Edit Alias:")
                TextField(
                    value = updatedTitle,
                    onValueChange = { updatedTitle = it },
                    modifier = Modifier.fillMaxWidth()
                )
            }
        },
        confirmButton = {
            Button(onClick = {
                onConfirm(person.copy(alias = updatedTitle.text))
            }) {
                Text("Update")
            }
        },
        dismissButton = {
            Button(onClick = { onDismiss() }) {
                Text("Cancel")
            }
        }
    )
}

