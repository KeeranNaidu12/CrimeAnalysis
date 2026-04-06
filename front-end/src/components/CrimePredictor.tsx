// app/components/CrimePredictor.tsx
'use client';

import React, { useState, useMemo } from 'react';
import Header from './Header';

interface PredictionResult {
    window_start: string;
    window_end: string;
    neighbourhood: string;
    neighbourhood_id: number;
    neighbourhood_clean: string;
    probability: number;
    predicted: boolean;
    risk_level: string;
}

interface PredictionSummary {
    total_windows: number;
    total_predictions: number;
    positive_predictions: number;
    positive_percentage: number;
    high_risk_alerts: number;
    medium_risk_alerts: number;
    low_risk_alerts: number;
}

interface PredictionResponse {
    crime_type: string;
    crime_display_name: string;
    date_range: {
        start: string;
        end: string;
    };
    neighbourhood_filter: string | null;
    summary: PredictionSummary;
    results: PredictionResult[];
    high_risk_alerts: PredictionResult[];
    neighbourhood_ranking: Array<{
        neighbourhood: string;
        avg_probability: number;
        predicted_count: number;
        total_windows: number;
    }>;
    window_summary: Array<{
        window: string;
        avg_risk: number;
        max_risk: number;
        predicted_count: number;
    }>;
}

export default function CrimePredictor() {
    const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedCrime, setSelectedCrime] = useState<string>('');
    const [selectedNeighborhood, setSelectedNeighborhood] = useState<string>('');
    const [startDate, setStartDate] = useState<string>('');
    const [endDate, setEndDate] = useState<string>('');
    const [activeTab, setActiveTab] = useState<'results' | 'alerts' | 'ranking'>('results');
    const [sortConfig, setSortConfig] = useState<{ key: keyof PredictionResult; direction: 'asc' | 'desc' } | null>(null);
    const [searchQuery, setSearchQuery] = useState<string>('');

    // Map frontend crime names to backend API crime types
    const getCrimeTypeForAPI = (crimeName: string): string => {
        const crimeMap: Record<string, string> = {
            'Assault': 'assault',
            'Auto Theft': 'auto_theft',
            'Break and Entering': 'break_and_enter',
            'Collision': 'collision'
        };
        return crimeMap[crimeName] || 'assault';
    };

    // Extract neighborhood ID from the display string
    const extractNeighborhoodId = (neighborhoodDisplay: string): number | null => {
        if (neighborhoodDisplay === 'All Neighborhoods') return null;
        const match = neighborhoodDisplay.match(/\((\d+)\)$/);
        return match ? parseInt(match[1]) : null;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!selectedCrime || !selectedNeighborhood || !startDate || !endDate) {
            setError('Please fill in all fields');
            return;
        }

        setLoading(true);
        setError(null);
        setPrediction(null);

        try {
            const crimeType = getCrimeTypeForAPI(selectedCrime);
            const neighborhoodId = extractNeighborhoodId(selectedNeighborhood);

            const requestBody = {
                crime_type: crimeType,
                start_date: startDate,
                end_date: endDate,
                neighbourhood_id: neighborhoodId
            };

            const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            const response = await fetch(`${apiUrl}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const data: PredictionResponse = await response.json();
            setPrediction(data);
            setActiveTab('results');
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
            console.error('Prediction error:', err);
        } finally {
            setLoading(false);
        }
    };

    // Sorting handler for table columns
    const handleSort = (key: keyof PredictionResult) => {
        if (!prediction) return;
        
        let direction: 'asc' | 'desc' = 'asc';
        if (sortConfig?.key === key && sortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setSortConfig({ key, direction });

        const sorted = [...prediction.results].sort((a, b) => {
            if (a[key] < b[key]) return direction === 'asc' ? -1 : 1;
            if (a[key] > b[key]) return direction === 'asc' ? 1 : -1;
            return 0;
        });
        setPrediction({ ...prediction, results: sorted });
    };

    // Filter results based on search query
    const filteredResults = useMemo(() => {
        if (!prediction || !searchQuery.trim()) return prediction?.results || [];
        
        const query = searchQuery.toLowerCase();
        return prediction.results.filter(result => 
            result.neighbourhood_clean.toLowerCase().includes(query) ||
            result.risk_level.toLowerCase().includes(query) ||
            result.neighbourhood_id.toString().includes(query)
        );
    }, [prediction, searchQuery]);

    // Professional risk indicator (text-based, no badges)
    const getRiskIndicator = (riskLevel: string): string => {
        switch (riskLevel) {
            case 'HIGH': return 'text-red-700 font-medium';
            case 'MEDIUM': return 'text-amber-700 font-medium';
            case 'LOW': return 'text-emerald-700 font-medium';
            default: return 'text-gray-600';
        }
    };

    // Probability color for subtle visual cues
    const getProbabilityStyle = (probability: number): string => {
        if (probability >= 0.7) return 'text-red-700';
        if (probability >= 0.4) return 'text-amber-700';
        return 'text-emerald-700';
    };

    // Format probability as percentage
    const formatProbability = (prob: number): string => `${(prob * 100).toFixed(1)}%`;

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
            <Header
                selectedCrime={selectedCrime}
                setSelectedCrime={setSelectedCrime}
                selectedNeighborhood={selectedNeighborhood}
                setSelectedNeighborhood={setSelectedNeighborhood}
                startDate={startDate}
                setStartDate={setStartDate}
                endDate={endDate}
                setEndDate={setEndDate}
                onSubmit={handleSubmit}
            />

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Loading State */}
                {loading && (
                    <div className="flex flex-col justify-center items-center py-16 animate-in fade-in duration-300">
                        <div className="w-10 h-10 border-2 border-blue-400 border-t-transparent rounded-full animate-spin mb-4"></div>
                        <p className="text-gray-600 font-['Book_Antiqua',_Palatino,_serif] text-lg">Processing prediction request</p>
                    </div>
                )}

                {/* Error State */}
                {error && !loading && (
                    <p className="text-red-700 font-medium font-['Book_Antiqua',_Palatino,_serif] text-center">{error}</p>
                )}

                {/* Results Display */}
                {prediction && !loading && !error && (
                    <div className="space-y-6 animate-in fade-in duration-500">
                        
                        {/* Summary Header */}
                        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                            <div>
                                <h2 className="text-xl font-['Book_Antiqua',_Palatino,_serif] text-gray-800 tracking-tight">
                                    {prediction.crime_display_name} · Risk Assessment
                                </h2>
                                <p className="text-gray-500 text-sm font-['Book_Antiqua',_Palatino,_serif] mt-1">
                                    {prediction.date_range.start} — {prediction.date_range.end}
                                    {prediction.neighbourhood_filter && ` · ${prediction.neighbourhood_filter}`}
                                </p>
                            </div>
                            <div className="flex items-center gap-6 text-sm font-['Book_Antiqua',_Palatino,_serif]">
                                <div className="text-right">
                                    <span className="text-gray-400">Positive Rate</span>
                                    <p className="text-lg font-medium text-gray-800">{prediction.summary.positive_percentage}%</p>
                                </div>
                                <div className="text-right">
                                    <span className="text-gray-400">Total Windows</span>
                                    <p className="text-lg font-medium text-gray-800">{prediction.summary.total_windows}</p>
                                </div>
                            </div>
                        </div>

                        {/* Tab Navigation */}
                        <div className="border-b border-gray-200">
                            <nav className="flex gap-8">
                                {(['results', 'alerts', 'ranking'] as const).map((tab) => (
                                    <button
                                        key={tab}
                                        onClick={() => { setActiveTab(tab); setSearchQuery(''); }}
                                        className={`pb-3 text-sm font-medium transition-all duration-200 font-['Book_Antiqua',_Palatino,_serif] ${
                                            activeTab === tab
                                                ? 'text-blue-600 border-b-2 border-blue-500'
                                                : 'text-gray-500 hover:text-gray-700'
                                        }`}
                                    >
                                        {tab === 'results' && 'Detailed Results'}
                                        {tab === 'alerts' && `High Risk · ${prediction.high_risk_alerts.length}`}
                                        {tab === 'ranking' && 'Neighborhood Ranking'}
                                    </button>
                                ))}
                            </nav>
                        </div>

                        {/* Search Bar - Only show on results tab */}
                        {activeTab === 'results' && (
                            <div className="flex items-center gap-3">
                                <div className="relative flex-1 max-w-md">
                                    <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                    </svg>
                                    <input
                                        type="text"
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                        placeholder="Search neighborhoods, risk levels, or IDs..."
                                        className="w-full pl-10 pr-4 py-2 text-sm bg-white/60 backdrop-blur-sm border border-gray-200 rounded-lg focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 transition-all font-['Book_Antiqua',_Palatino,_serif]"
                                    />
                                </div>
                                {searchQuery && (
                                    <button
                                        onClick={() => setSearchQuery('')}
                                        className="text-sm text-gray-500 hover:text-gray-700 transition-colors font-['Book_Antiqua',_Palatino,_serif]"
                                    >
                                        Clear
                                    </button>
                                )}
                                <span className="text-xs text-gray-400 font-['Book_Antiqua',_Palatino,_serif]">
                                    {filteredResults.length} result{filteredResults.length !== 1 ? 's' : ''}
                                </span>
                            </div>
                        )}

                        {/* Tab Content Area */}
                        <div className="min-h-[400px]">
                            
                            {/* RESULTS TAB - Professional Table View */}
                            {activeTab === 'results' && (
                                <div className="overflow-x-auto">
                                    <table className="w-full">
                                        <thead className="border-b border-gray-200">
                                            <tr className="text-left text-xs uppercase tracking-wide text-gray-500 font-['Book_Antiqua',_Palatino,_serif]">
                                                <th 
                                                    className="py-4 px-5 cursor-pointer hover:text-gray-700 transition-colors"
                                                    onClick={() => handleSort('neighbourhood_clean')}
                                                >
                                                    Neighborhood
                                                </th>
                                                <th className="py-4 px-5">Time Window</th>
                                                <th 
                                                    className="py-4 px-5 cursor-pointer hover:text-gray-700 transition-colors"
                                                    onClick={() => handleSort('risk_level')}
                                                >
                                                    Risk Level
                                                </th>
                                                <th 
                                                    className="py-4 px-5 text-right cursor-pointer hover:text-gray-700 transition-colors"
                                                    onClick={() => handleSort('probability')}
                                                >
                                                    Probability
                                                </th>
                                                <th className="py-4 px-5 text-right">ID</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-100">
                                            {filteredResults.slice(0, 100).map((result, index) => (
                                                <tr 
                                                    key={`${result.neighbourhood_id}-${result.window_start}-${index}`}
                                                    className="hover:bg-gradient-to-r hover:from-blue-50/50 hover:to-purple-50/50 transition-all duration-200 font-['Book_Antiqua',_Palatino,_serif] hover:shadow-sm hover:shadow-blue-100/50 animate-pulse-hover"
                                                >
                                                    <td className="py-4 px-5 text-gray-800">
                                                        {result.neighbourhood_clean}
                                                    </td>
                                                    <td className="py-4 px-5 text-gray-600 text-sm">
                                                        {result.window_start}<br />
                                                        <span className="text-gray-400">to {result.window_end}</span>
                                                    </td>
                                                    <td className="py-4 px-5">
                                                        <span className={getRiskIndicator(result.risk_level)}>
                                                            {result.risk_level}
                                                        </span>
                                                    </td>
                                                    <td className="py-4 px-5 text-right">
                                                        <span className={getProbabilityStyle(result.probability)}>
                                                            {formatProbability(result.probability)}
                                                        </span>
                                                    </td>
                                                    <td className="py-4 px-5 text-right text-gray-400 text-sm">
                                                        {result.neighbourhood_id}
                                                    </td>
                                                </tr>
                                            ))}
                                            {filteredResults.length === 0 && (
                                                <tr>
                                                    <td colSpan={5} className="py-8 text-center text-gray-400 font-['Book_Antiqua',_Palatino,_serif]">
                                                        {searchQuery ? `No results match "${searchQuery}"` : 'No results match the selected criteria'}
                                                    </td>
                                                </tr>
                                            )}
                                        </tbody>
                                    </table>
                                    {filteredResults.length > 100 && (
                                        <p className="py-3 text-sm text-gray-500 text-center font-['Book_Antiqua',_Palatino,_serif]">
                                            Showing first 100 of {filteredResults.length} results
                                        </p>
                                    )}
                                </div>
                            )}

                            {/* ALERTS TAB - Clean List View */}
                            {activeTab === 'alerts' && (
                                <div className="overflow-x-auto">
                                    {prediction.high_risk_alerts.length > 0 ? (
                                        <table className="w-full">
                                            <thead className="border-b border-gray-200">
                                                <tr className="text-left text-xs uppercase tracking-wide text-gray-500 font-['Book_Antiqua',_Palatino,_serif]">
                                                    <th className="py-4 px-5">Neighborhood</th>
                                                    <th className="py-4 px-5">Time Window</th>
                                                    <th className="py-4 px-5 text-right">Probability</th>
                                                    <th className="py-4 px-5 text-right">Risk</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-gray-100">
                                                {prediction.high_risk_alerts.map((alert, index) => (
                                                    <tr 
                                                        key={`alert-${index}`}
                                                        className="hover:bg-gradient-to-r hover:from-red-50/30 hover:to-amber-50/30 transition-all duration-200 font-['Book_Antiqua',_Palatino,_serif] hover:shadow-sm hover:shadow-red-100/30 animate-pulse-hover"
                                                    >
                                                        <td className="py-4 px-5 text-gray-800 font-medium">
                                                            {alert.neighbourhood_clean}
                                                        </td>
                                                        <td className="py-4 px-5 text-gray-600 text-sm">
                                                            {alert.window_start}<br />
                                                            <span className="text-gray-400">to {alert.window_end}</span>
                                                        </td>
                                                        <td className="py-4 px-5 text-right">
                                                            <span className="text-red-700 font-medium">
                                                                {formatProbability(alert.probability)}
                                                            </span>
                                                        </td>
                                                        <td className="py-4 px-5 text-right">
                                                            <span className="text-red-700 font-medium">HIGH</span>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    ) : (
                                        <p className="py-4 text-gray-500 font-['Book_Antiqua',_Palatino,_serif]">
                                            No high-risk alerts detected for the selected parameters
                                        </p>
                                    )}
                                </div>
                            )}

                            {/* RANKING TAB - Sorted Table */}
                            {activeTab === 'ranking' && (
                                <div className="overflow-x-auto">
                                    <table className="w-full">
                                        <thead className="border-b border-gray-200">
                                            <tr className="text-left text-xs uppercase tracking-wide text-gray-500 font-['Book_Antiqua',_Palatino,_serif]">
                                                <th className="py-4 px-5">Rank</th>
                                                <th className="py-4 px-5">Neighborhood</th>
                                                <th className="py-4 px-5 text-right">Avg. Probability</th>
                                                <th className="py-4 px-5 text-right">Predicted</th>
                                                <th className="py-4 px-5 text-right">Total Windows</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-100">
                                            {prediction.neighbourhood_ranking.map((item, index) => (
                                                <tr 
                                                    key={`rank-${index}`}
                                                    className="hover:bg-gradient-to-r hover:from-blue-50/50 hover:to-purple-50/50 transition-all duration-200 font-['Book_Antiqua',_Palatino,_serif] hover:shadow-sm hover:shadow-blue-100/50 animate-pulse-hover"
                                                >
                                                    <td className="py-4 px-5 text-gray-500">#{index + 1}</td>
                                                    <td className="py-4 px-5 text-gray-800">{item.neighbourhood}</td>
                                                    <td className="py-4 px-5 text-right">
                                                        <span className={getProbabilityStyle(item.avg_probability)}>
                                                            {formatProbability(item.avg_probability)}
                                                        </span>
                                                    </td>
                                                    <td className="py-4 px-5 text-right text-gray-600">{item.predicted_count}</td>
                                                    <td className="py-4 px-5 text-right text-gray-400">{item.total_windows}</td>
                                                </tr>
                                            ))}
                                            {prediction.neighbourhood_ranking.length === 0 && (
                                                <tr>
                                                    <td colSpan={5} className="py-8 text-center text-gray-400 font-['Book_Antiqua',_Palatino,_serif]">
                                                        No ranking data available
                                                    </td>
                                                </tr>
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Initial Empty State - Clean text on page, no card block */}
                {!prediction && !loading && !error && (
                    <div className="py-12 text-center">
                        <h3 className="text-lg font-['Book_Antiqua',_Palatino,_serif] text-gray-800 mb-2">
                            Configure Prediction Parameters
                        </h3>
                        <p className="text-gray-500 font-['Book_Antiqua',_Palatino,_serif] max-w-md mx-auto">
                            Select a crime type, neighborhood, and date range above to generate an AI-powered risk assessment for Toronto neighborhoods.
                        </p>
                    </div>
                )}
            </main>

            {/* Global styles for consistent animations */}
            <style jsx global>{`
                @keyframes fade-in {
                    from { opacity: 0; transform: translateY(4px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                @keyframes pulse-hover {
                    0%, 100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
                    50% { box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.08); }
                }
                .animate-in {
                    animation: fade-in 300ms ease-out forwards;
                }
                .animate-pulse-hover:hover {
                    animation: pulse-hover 1.5s ease-in-out infinite;
                }
                /* Smooth transition for search filter */
                tbody tr {
                    transition: background-color 150ms ease, box-shadow 150ms ease;
                }
            `}</style>
        </div>
    );
}