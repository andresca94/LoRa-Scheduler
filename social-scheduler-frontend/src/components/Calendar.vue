<template>
    <div class="calendar">
      <h2>Select a Day</h2>
      <div class="days">
        <button v-for="day in days" :key="day" @click="selectDate(day)">
          {{ day }}
        </button>
      </div>
      <p>Selected Date: {{ selectedDate }}</p>
    </div>
  </template>
  
  <script>
  import { ref, defineEmits } from 'vue';
  
  export default {
    setup(_, { emit }) {
      const days = Array.from({ length: 30 }, (_, i) => i + 1);
      const selectedDate = ref('');
  
      const selectDate = (day) => {
        const today = new Date();
        selectedDate.value = `${today.getFullYear()}-${today.getMonth() + 1}-${day}`;
        emit('select-date', selectedDate.value);
      };
  
      return { days, selectedDate, selectDate };
    },
  };
  </script>
  
  <style>
  .calendar button {
    margin: 5px;
    padding: 10px;
    cursor: pointer;
  }
  </style>
  