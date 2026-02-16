package com.kotlintestapp.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import com.kotlintestapp.data.models.ApiResult
import com.kotlintestapp.data.models.User
import com.kotlintestapp.ui.viewmodels.HomeViewModel

@Composable
fun HomeScreen(viewModel: HomeViewModel) {
    val userState by viewModel.users.collectAsState()

    Column {
        Text(text = "Users List")
        
        when (val result = userState) {
            is ApiResult.Loading -> Text("Loading...")
            is ApiResult.Error -> Text("Error: ${result.exception.message}")
            is ApiResult.Success -> UserList(users = result.data)
        }
    }
}

@Composable
fun UserList(users: List<User>) {
    LazyColumn {
        items(users) { user ->
            UserRow(user)
        }
    }
}

@Composable
fun UserRow(user: User) {
    Text(text = "${user.name} (${user.role})")
}
