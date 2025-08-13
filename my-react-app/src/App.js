import { database } from './firebase';
import { ref, onValue } from 'firebase/database';
import { useState, useEffect } from 'react';

function App() {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    const logsRef = ref(database, 'material_logs');
    onValue(logsRef, (snapshot) => {
      const data = snapshot.val();
      if (data) {
        const logArray = Object.values(data);
        setLogs(logArray);
      }
    });
  }, []);

  return (
    <div>
      <h1>Material Logs</h1>
      <ul>
        {logs.map((log, index) => (
          <li key={index}>
            {log.timestamp}: {log.material} ({log.rx_sources.join(', ')})
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;