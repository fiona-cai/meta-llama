import React, { useEffect, useState } from 'react';

function App() {
    const [data, setData] = useState(null);

    useEffect(() => {
        fetch('/api')
            .then(response => response.json())
            .then(data => setData(data.message));
    }, []);

    return (
        <div>
            <h1>{data || "Loading..."}</h1>
        </div>
    );
}

export default App;
