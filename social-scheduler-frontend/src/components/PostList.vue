<template>
    <div>
      <h2>Scheduled Posts</h2>
      <div v-if="posts.length === 0">No posts scheduled yet.</div>
      <ul>
        <li v-for="post in posts" :key="post">
          <img :src="`http://localhost:8000/${post}`" width="100" />
        </li>
      </ul>
    </div>
  </template>
  
  <script>
  import { defineProps } from 'vue';
  
  export default {
    props: ['posts']
  };
  </script>
  