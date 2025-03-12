import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './App.css'

function App() {
  const [augments, setAugments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [minGames, setMinGames] = useState(1);
  const [sortConfig, setSortConfig] = useState({ key: 'avg_placement', direction: 'asc' });
  const [error, setError] = useState(null);

  // API base URL - get from environment variables or use localhost as fallback
  const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

  useEffect(() => {
    fetchAugments();
  }, []);

  const fetchAugments = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/augments?min_games=${minGames}&complete_only=true`);
      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }
      const data = await response.json();
      setAugments(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching augment data:', err);
      setError('Failed to load augment data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
  };

  const handleMinGamesChange = (e) => {
    const value = parseInt(e.target.value) || 1;
    setMinGames(Math.max(1, value));
  };

  const requestSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const getPositionAvg = (item, position) => {
    return item.position_data[position]?.avg_placement || '-';
  };

  const getSortedData = () => {
    const filteredData = augments.filter((item) => 
      item.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    if (!sortConfig.key) return filteredData;

    return [...filteredData].sort((a, b) => {
      let aValue, bValue;
      
      if (sortConfig.key.startsWith('pos')) {
        const position = sortConfig.key.slice(3);
        aValue = a.position_data[position]?.avg_placement ?? Number.MAX_SAFE_INTEGER;
        bValue = b.position_data[position]?.avg_placement ?? Number.MAX_SAFE_INTEGER;
      } else {
        aValue = a[sortConfig.key];
        bValue = b[sortConfig.key];
      }
      
      if (aValue === bValue) {
        return 0;
      }
      
      const direction = sortConfig.direction === 'asc' ? 1 : -1;
      return aValue < bValue ? -1 * direction : 1 * direction;
    });
  };

  const getSortIndicator = (key) => {
    if (sortConfig.key === key) {
      return sortConfig.direction === 'asc' ? ' 🔼' : ' 🔽';
    }
    return '';
  };

  const sortedData = getSortedData();

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center p-8 bg-red-50 rounded-lg border border-red-200 max-w-md">
          <h2 className="text-xl font-semibold text-red-700 mb-4">Error</h2>
          <p className="text-gray-700">{error}</p>
          <button 
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
            onClick={fetchAugments}
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <h1 className="text-3xl font-bold text-center mb-6">TFT Augment Analysis</h1>
      
      <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="flex flex-col">
          <label htmlFor="search" className="mb-1 font-medium">Search Augments:</label>
          <input 
            id="search"
            type="text" 
            value={searchTerm} 
            onChange={handleSearch} 
            className="p-2 border rounded"
            placeholder="Enter augment name..."
          />
        </div>
        
        <div className="flex flex-col">
          <label htmlFor="minGames" className="mb-1 font-medium">Minimum Games Played:</label>
          <div className="flex">
            <input 
              id="minGames"
              type="number" 
              value={minGames} 
              onChange={handleMinGamesChange}
              className="p-2 border rounded w-24 mr-2"
              min="1"
            />
            <button 
              onClick={fetchAugments}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Apply
            </button>
          </div>
        </div>
      </div>
      
      {loading ? (
        <div className="text-center p-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent"></div>
          <p className="mt-2">Loading augment data...</p>
        </div>
      ) : (
        <>
          <div className="mb-4 text-sm text-gray-600">
            Showing {sortedData.length} augments (filtered from {augments.length} total)
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse">
              <thead className="bg-gray-100">
                <tr>
                  <th 
                    className="p-3 border border-gray-300 cursor-pointer hover:bg-gray-200"
                    onClick={() => requestSort('name')}
                  >
                    Augment Name{getSortIndicator('name')}
                  </th>
                  <th 
                    className="p-3 border border-gray-300 cursor-pointer hover:bg-gray-200"
                    onClick={() => requestSort('avg_placement')}
                  >
                    Average Placement{getSortIndicator('avg_placement')}
                  </th>
                  <th 
                    className="p-3 border border-gray-300 cursor-pointer hover:bg-gray-200"
                    onClick={() => requestSort('total_games')}
                  >
                    Games Played{getSortIndicator('total_games')}
                  </th>
                  <th 
                    className="p-3 border border-gray-300 cursor-pointer hover:bg-gray-200"
                    onClick={() => requestSort('pos0')}
                  >
                    Avg At 2-1{getSortIndicator('pos0')}
                  </th>
                  <th 
                    className="p-3 border border-gray-300 cursor-pointer hover:bg-gray-200"
                    onClick={() => requestSort('pos1')}
                  >
                    Avg At 3-2{getSortIndicator('pos1')}
                  </th>
                  <th 
                    className="p-3 border border-gray-300 cursor-pointer hover:bg-gray-200"
                    onClick={() => requestSort('pos2')}
                  >
                    Avg At 4-2{getSortIndicator('pos2')}
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedData.map((augment) => (
                  <tr key={augment.name} className="hover:bg-gray-50">
                    <td className="p-3 border border-gray-300">{augment.name}</td>
                    <td className="p-3 border border-gray-300 text-center">{augment.avg_placement.toFixed(2)}</td>
                    <td className="p-3 border border-gray-300 text-center">{augment.total_games}</td>
                    <td className="p-3 border border-gray-300 text-center">{getPositionAvg(augment, '0')}</td>
                    <td className="p-3 border border-gray-300 text-center">{getPositionAvg(augment, '1')}</td>
                    <td className="p-3 border border-gray-300 text-center">{getPositionAvg(augment, '2')}</td>
                  </tr>
                ))}
                {sortedData.length === 0 && (
                  <tr>
                    <td colSpan="6" className="p-4 text-center text-gray-500">
                      No augments found matching your search criteria
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

export default App;