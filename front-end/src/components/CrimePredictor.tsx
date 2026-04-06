// app/components/CrimePredictor.tsx
'use client';

import React, { useState } from 'react';
import Header from './Header'

interface CrimePredictionResponse {
    predictedRisk: number;
    confidence: number;
    contributingFactors: string[];
}

export default function CrimePredictor() {
    const [prediction, setPrediction] = useState<CrimePredictionResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [selectedCrime, setSelectedCrime] = useState<string>('');
    const [selectedNeighborhood, setSelectedNeighborhood] = useState<string>('');
    const [startDate, setStartDate] = useState<string>('');
    const [endDate, setEndDate] = useState<string>('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!selectedCrime || !selectedNeighborhood || !startDate || !endDate) {
            alert('Please fill in all fields');
            return;
        }

        setLoading(true);

        // Simulate API call - replace with your actual API endpoint
        setTimeout(() => {
            const mockPrediction: CrimePredictionResponse = {
                predictedRisk: Math.random() * 100,
                confidence: 70 + Math.random() * 25,
                contributingFactors: [
                    'Historical crime patterns',
                    'Time of year correlation',
                    'Neighborhood demographics',
                    'Recent incident frequency'
                ]
            };
            setPrediction(mockPrediction);
            setLoading(false);
        }, 1500);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
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

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                {loading && (
                    <div className="flex justify-center items-center py-20">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                    </div>
                )}

                {prediction && !loading && (
                    <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
                        <h2 className="text-2xl font-bold text-gray-900 mb-6">Prediction Results</h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div>
                                <div className="bg-blue-50 rounded-xl p-6">
                                    <p className="text-sm text-blue-600 font-semibold mb-2">Predicted Risk Level</p>
                                    <p className="text-4xl font-bold text-gray-900">{prediction.predictedRisk.toFixed(1)}%</p>
                                    <div className="mt-4 h-2 bg-blue-200 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-blue-600 rounded-full transition-all duration-500"
                                            style={{ width: `${prediction.predictedRisk}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div className="bg-green-50 rounded-xl p-6">
                                    <p className="text-sm text-green-600 font-semibold mb-2">Model Confidence</p>
                                    <p className="text-4xl font-bold text-gray-900">{prediction.confidence.toFixed(1)}%</p>
                                    <div className="mt-4 h-2 bg-green-200 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-green-600 rounded-full transition-all duration-500"
                                            style={{ width: `${prediction.confidence}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="mt-8">
                            <h3 className="text-lg font-semibold text-gray-900 mb-4">Contributing Factors</h3>
                            <div className="flex flex-wrap gap-2">
                                {prediction.contributingFactors.map((factor, index) => (
                                    <span key={index} className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm">
                                        {factor}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}