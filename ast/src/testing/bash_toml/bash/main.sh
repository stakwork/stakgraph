#!/usr/bin/env bash

fetch_user() {
  local user_id="$1"
  echo "fetching ${user_id}"
}

run() {
  fetch_user "42"
}

run
