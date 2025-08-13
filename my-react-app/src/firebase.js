import { initializeApp } from 'firebase/app';
import { getAnalytics } from 'firebase/analytics';
import { getDatabase } from 'firebase/database';

const firebaseConfig = {
  apiKey: "AIzaSyApin9R_LCeiIaltDV1cyGpOS_saFH4UZc",
  authDomain: "homesecurity-4731c.firebaseapp.com",
  databaseURL: "https://homesecurity-4731c-default-rtdb.firebaseio.com",
  projectId: "homesecurity-4731c",
  storageBucket: "homesecurity-4731c.firebasestorage.app",
  messagingSenderId: "954675880486",
  appId: "1:954675880486:web:bcbea3ecbaef34d1f8fce7",
  measurementId: "G-RM52NW1N66"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const database = getDatabase(app);

export { database };