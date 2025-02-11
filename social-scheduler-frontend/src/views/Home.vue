<template>
  <div>
    <h2>Select a Day</h2>
    <div class="calendar">
      <button 
        v-for="day in daysInMonth" 
        :key="day" 
        @click="setSelectedDate(day)"
        :class="{ active: selectedDate === formatDate(day) }">
        {{ day }}
      </button>
    </div>
    <p>📅 Selected Date: <strong>{{ selectedDate || "None" }}</strong></p>

    <h2>Schedule a Post</h2>
    <input v-model="topic" placeholder="Enter a topic..." />
    <button @click="schedulePost" :disabled="loading">Schedule</button>

    <!-- Show progress loader while scheduling -->
    <div v-if="loading" class="loader"></div>

    <p v-if="errorMessage" class="error">{{ errorMessage }}</p>

    <h2>Scheduled Posts</h2>
    <ul>
      <li v-for="post in scheduledPosts" :key="post.date">
        📅 Date: {{ post.date }} | 📝 Topic: {{ post.topic }} | {{ post.status }}
        <div v-if="post.generated_image">
          <img :src="'http://127.0.0.1:8000/images/' + post.generated_image" alt="Generated Image" class="generated-img"/>
        </div>
      </li>
    </ul>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from "vue";
import axios from "axios";

// State variables
const selectedDate = ref(null);
const topic = ref("");
const errorMessage = ref("");
const scheduledPosts = ref([]);
const loading = ref(false);

// Dynamically get days of the current month
const daysInMonth = computed(() => {
  const today = new Date();
  const year = today.getFullYear();
  const month = today.getMonth() + 1;
  const days = new Date(year, month, 0).getDate(); // Get last day of the month
  return Array.from({ length: days }, (_, i) => i + 1);
});

// Format date function
const formatDate = (day) => {
  const today = new Date();
  const year = today.getFullYear();
  const month = today.getMonth() + 1;
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
};

// Select a date
const setSelectedDate = (day) => {
  selectedDate.value = formatDate(day);
  errorMessage.value = "";
};

// Schedule post request
const schedulePost = async () => {
  if (!selectedDate.value || !topic.value.trim()) {
    errorMessage.value = "⚠️ Select a date and enter a topic first!";
    return;
  }

  loading.value = true; // Show loader

  try {
    const response = await axios.post("http://127.0.0.1:8000/schedule_post/", {
      date: selectedDate.value,
      topic: topic.value,
    });

    if (response.status === 200) {
      alert("✅ Post scheduled successfully!");
      topic.value = "";
      fetchPosts();
    }
  } catch (error) {
    errorMessage.value = "❌ Failed to schedule post. Try again!";
  } finally {
    loading.value = false; // Hide loader
  }
};

// Fetch all scheduled posts
const fetchPosts = async () => {
  try {
    const response = await axios.get("http://127.0.0.1:8000/list_scheduled_posts/");
    scheduledPosts.value = response.data.scheduled_posts;
  } catch (error) {
    console.error("❌ Error fetching posts:", error);
  }
};

// Fetch posts on component mount
onMounted(() => {
  fetchPosts();
});
</script>

<style scoped>
.calendar {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-bottom: 10px;
}

.calendar button {
  padding: 5px 10px;
  cursor: pointer;
  border: none;
  background: #ddd;
  border-radius: 5px;
}

.calendar button.active {
  background: #007bff;
  color: white;
}

button {
  padding: 8px 12px;
  margin-top: 5px;
  border: none;
  background-color: #007bff;
  color: white;
  cursor: pointer;
  border-radius: 5px;
}

button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.error {
  color: red;
  font-weight: bold;
}

/* Loader (progress circle) */
.loader {
  margin: 10px auto;
  width: 40px;
  height: 40px;
  border: 4px solid rgba(0, 0, 0, 0.1);
  border-top: 4px solid #007bff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.generated-img {
  max-width: 100px;
  margin-top: 10px;
  border-radius: 5px;
}
</style>
