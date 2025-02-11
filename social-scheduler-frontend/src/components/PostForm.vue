<template>
    <div class="post-form">
      <h2>Schedule a Post</h2>
      <input v-model="topic" placeholder="Enter a topic..." />
      <button @click="schedulePost">Schedule</button>
    </div>
  </template>
  
  <script>
  import { ref, defineProps, defineEmits } from 'vue';
  import axios from 'axios';
  
  export default {
    props: ['selectedDate'],
    setup(props, { emit }) {
      const topic = ref('');
  
      const schedulePost = async () => {
        if (!props.selectedDate || !topic.value) {
          alert("Select a date and enter a topic first!");
          return;
        }
  
        try {
          // Trigger scraping
          await axios.get(`http://localhost:8000/scrape_images/?keyword=${topic.value}`);
  
          // Process images
          await axios.post('http://localhost:8000/process_images/');
  
          // Generate captions
          await axios.post('http://localhost:8000/generate_captions/');
  
          // Train LoRA model (optional step)
          await axios.post('http://localhost:8000/train_lora/');
  
          // Generate image using fine-tuned model
          const response = await axios.post('http://localhost:8000/generate/', {
            prompt: topic.value,
            num_steps: 30,
            seed: 42
          });
  
          console.log(response.data);
          emit('post-scheduled');
        } catch (error) {
          console.error("Error scheduling post", error);
        }
      };
  
      return { topic, schedulePost };
    },
  };
  </script>
  