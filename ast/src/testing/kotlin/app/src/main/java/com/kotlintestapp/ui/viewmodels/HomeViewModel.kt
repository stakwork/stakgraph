package com.kotlintestapp.ui.viewmodels

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.kotlintestapp.data.models.ApiResult
import com.kotlintestapp.data.models.User
import com.kotlintestapp.data.repository.UserRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class HomeViewModel @Inject constructor(
    private val userRepository: UserRepository
) : ViewModel() {

    private val _users = MutableStateFlow<ApiResult<List<User>>>(ApiResult.Loading)
    val users: StateFlow<ApiResult<List<User>>> = _users.asStateFlow()

    init {
        fetchUsers()
    }

    fun fetchUsers() {
        viewModelScope.launch {
            userRepository.getUsers().collect { result ->
                _users.value = result
            }
        }
    }

    fun onUserClicked(user: User) {
        // Handle user click
    }
}
