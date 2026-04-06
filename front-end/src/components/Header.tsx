// app/components/Header.tsx
'use client';

import React, { useState, useRef, useEffect } from 'react';

interface HeaderProps {
    selectedCrime: string;
    setSelectedCrime: (crime: string) => void;
    selectedNeighborhood: string;
    setSelectedNeighborhood: (neighborhood: string) => void;
    startDate: string;
    setStartDate: (date: string) => void;
    endDate: string;
    setEndDate: (date: string) => void;
    onSubmit: (e: React.FormEvent) => void;
}

const crimeTypes = ['Assault', 'Auto Theft', 'Break and Entering', 'Collision'];

const neighborhoods = ["All Neighborhoods",
    "Agincourt North (129)", "Agincourt South-Malvern West (128)", "Alderwood (20)",
    "Annex (95)", "Avondale (153)", "Banbury-Don Mills (42)", "Bathurst Manor (34)",
    "Bay-Cloverhill (169)", "Bayview Village (52)", "Bayview Woods-Steeles (49)",
    "Bedford Park-Nortown (39)", "Beechborough-Greenbrook (112)", "Bendale-Glen Andrew (156)",
    "Bendale South (157)", "Birchcliffe-Cliffside (122)", "Black Creek (24)",
    "Blake-Jones (69)", "Briar Hill-Belgravia (108)", "Bridle Path-Sunnybrook-York Mills (41)",
    "Broadview North (57)", "Brookhaven-Amesbury (30)", "Cabbagetown-South St.James Town (71)",
    "Caledonia-Fairbank (109)", "Casa Loma (96)", "Centennial Scarborough (133)",
    "Church-Wellesley (167)", "Clairlea-Birchmount (120)", "Clanton Park (33)",
    "Cliffcrest (123)", "Corso Italia-Davenport (92)", "Danforth (66)",
    "Danforth East York (59)", "Don Valley Village (47)", "Dorset Park (126)",
    "Dovercourt Village (172)", "Downsview (155)", "Downtown Yonge East (168)",
    "Dufferin Grove (83)", "East End-Danforth (62)", "East L'Amoreaux (148)",
    "East Willowdale (152)", "Edenbridge-Humber Valley (9)", "Eglinton East (138)",
    "Elms-Old Rexdale (5)", "Englemount-Lawrence (32)", "Eringate-Centennial-West Deane (11)",
    "Etobicoke City Centre (159)", "Etobicoke West Mall (13)", "Fenside-Parkwoods (150)",
    "Flemingdon Park (44)", "Forest Hill North (102)", "Forest Hill South (101)",
    "Fort York-Liberty Village (163)", "Glenfield-Jane Heights (25)", "Golfdale-Cedarbrae-Woburn (141)",
    "Greenwood-Coxwell (65)", "Guildwood (140)", "Harbourfront-CityPlace (165)",
    "Henry Farm (53)", "High Park-Swansea (87)", "High Park North (88)",
    "Highland Creek (134)", "Hillcrest Village (48)", "Humber Bay Shores (161)",
    "Humber Heights-Westmount (8)", "Humber Summit (21)", "Humbermede (22)",
    "Humewood-Cedarvale (106)", "Ionview (125)", "Islington (158)",
    "Junction-Wallace Emerson (171)", "Junction Area (90)", "Keelesdale-Eglinton West (110)",
    "Kennedy Park (124)", "Kensington-Chinatown (78)", "Kingsview Village-The Westway (6)",
    "Kingsway South (15)", "L'Amoreaux West (147)", "Lambton Baby Point (114)",
    "Lansing-Westgate (38)", "Lawrence Park North (105)", "Lawrence Park South (103)",
    "Leaside-Bennington (56)", "Little Portugal (84)", "Long Branch (19)",
    "Malvern East (146)", "Malvern West (145)", "Maple Leaf (29)", "Markland Wood (12)",
    "Milliken (130)", "Mimico-Queensway (160)", "Morningside (135)", "Morningside Heights (144)",
    "Moss Park (73)", "Mount Dennis (115)", "Mount Olive-Silverstone-Jamestown (2)",
    "Mount Pleasant East (99)", "New Toronto (18)", "Newtonbrook East (50)",
    "Newtonbrook West (36)", "North Riverdale (68)", "North St.James Town (74)",
    "North Toronto (173)", "O'Connor-Parkview (54)", "Oakdale-Beverley Heights (154)",
    "Oakridge (121)", "Oakwood Village (107)", "Old East York (58)", "Palmerston-Little Italy (80)",
    "Parkwoods-O'Connor Hills (149)", "Pelmo Park-Humberlea (23)", "Playter Estates-Danforth (67)",
    "Pleasant View (46)", "Princess-Rosethorn (10)", "Regent Park (72)", "Rexdale-Kipling (4)",
    "Rockcliffe-Smythe (111)", "Roncesvalles (86)", "Rosedale-Moore Park (98)",
    "Runnymede-Bloor West Village (89)", "Rustic (28)", "Scarborough Village (139)",
    "South Eglinton-Davisville (174)", "South Parkdale (85)", "South Riverdale (70)",
    "St Lawrence-East Bayfront-The Islands (166)", "St.Andrew-Windfields (40)", "Steeles (116)",
    "Stonegate-Queensway (16)", "Tam O'Shanter-Sullivan (118)", "Taylor-Massey (61)",
    "The Beaches (63)", "Thistletown-Beaumond Heights (3)", "Thorncliffe Park (55)",
    "Trinity-Bellwoods (81)", "University (79)", "Victoria Village (43)", "Wellington Place (164)",
    "West Hill (136)", "West Humber-Clairville (1)", "West Queen West (162)", "West Rouge (143)",
    "Westminster-Branson (35)", "Weston-Pelham Park (91)", "Weston (113)", "Wexford/Maryvale (119)",
    "Willowdale West (37)", "Willowridge-Martingrove-Richview (7)", "Woburn North (142)",
    "Woodbine-Lumsden (60)", "Woodbine Corridor (64)", "Wychwood (94)", "Yonge-Bay Corridor (170)",
    "Yonge-Doris (151)", "Yonge-Eglinton (100)", "Yonge-St.Clair (97)", "York University Heights (27)",
    "Yorkdale-Glen Park (31)"
];

// Enhanced Dropdown Component with overlay-style selection
function EnhancedDropdown({ 
    value, 
    onChange, 
    placeholder, 
    label,
    options
}: { 
    value: string; 
    onChange: (val: string) => void; 
    placeholder: string; 
    label: string;
    options: string[];
}) {
    const [isOpen, setIsOpen] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const dropdownRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
                setSearchTerm('');
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []); // Empty dependency array - no issues

    const filteredOptions = options.filter(option =>
        option.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const handleSelect = (option: string) => {
        onChange(option);
        setIsOpen(false);
        setSearchTerm('');
    };

    const displayValue = value || placeholder;

    return (
        <div className="flex flex-col" ref={dropdownRef}>
            <label className="text-xs uppercase tracking-wider text-gray-400 mb-1 font-medium">
                {label}
            </label>
            <div className="relative">
                <button
                    type="button"
                    onClick={() => setIsOpen(!isOpen)}
                    className="w-full text-left text-lg py-1 border-b border-transparent hover:border-blue-300 transition-all duration-200 group flex items-center justify-between text-gray-800"
                >
                    <span className={!value ? 'opacity-50' : 'opacity-100'}>
                        {displayValue}
                    </span>
                    <svg 
                        className={`w-4 h-4 transition-transform duration-200 text-gray-400 ${isOpen ? 'rotate-180' : ''}`}
                        fill="none" 
                        stroke="currentColor" 
                        viewBox="0 0 24 24"
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                </button>

                {/* Overlay-style dropdown */}
                {isOpen && (
                    <>
                        {/* Backdrop overlay */}
                        <div className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm" onClick={() => setIsOpen(false)} />
                        
                        {/* Dropdown panel - squared corners */}
                        <div className="absolute z-50 left-0 right-0 mt-2 bg-white shadow-2xl border border-gray-100 overflow-hidden animate-in slide-in-from-top-2 duration-200">
                            {/* Search input */}
                            <div className="p-3 border-b border-gray-100">
                                <input
                                    type="text"
                                    autoFocus
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    placeholder={`Search ${label.toLowerCase()}...`}
                                    className="w-full px-3 py-2 text-sm bg-gray-50 border border-gray-200 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 transition-all"
                                />
                            </div>
                            
                            {/* Options list */}
                            <div className="max-h-64 overflow-y-auto">
                                {filteredOptions.length > 0 ? (
                                    filteredOptions.map((option) => (
                                        <button
                                            key={option}
                                            type="button"
                                            onClick={() => handleSelect(option)}
                                            className={`w-full text-left px-4 py-2.5 text-sm transition-colors duration-150 hover:bg-gradient-to-r hover:from-blue-50 hover:to-purple-50 ${
                                                option === value ? 'bg-blue-50 text-blue-700 font-medium' : 'text-gray-700'
                                            }`}
                                        >
                                            {option}
                                        </button>
                                    ))
                                ) : (
                                    <div className="px-4 py-8 text-center text-gray-400 text-sm">
                                        No matching options found
                                    </div>
                                )}
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

// Date validation helper
const validateAndFormatDate = (dateString: string): string => {
    if (!dateString) return '';
    
    const date = new Date(dateString);
    const year = date.getFullYear();
    
    // Check if year is between 2014 and 2030
    if (year >= 2014 && year <= 2030 && !isNaN(date.getTime())) {
        return dateString;
    }
    
    return '';
};

// Custom Date Input Component with enhanced scroll sensitivity
function DateInput({ 
    value, 
    onChange, 
    placeholder, 
    label
}: { 
    value: string; 
    onChange: (val: string) => void; 
    placeholder: string; 
    label: string;
}) {
    const [isEditing, setIsEditing] = useState(false);
    const [localValue, setLocalValue] = useState(value);
    const inputRef = useRef<HTMLInputElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    // Separate useEffect for focus management - stable dependency
    useEffect(() => {
        if (isEditing && inputRef.current) {
            inputRef.current.focus();
            // Trigger the native date picker
            if (inputRef.current.showPicker) {
                inputRef.current.showPicker();
            }
        }
    }, [isEditing]); // Only depends on isEditing

    const handleBlur = () => {
        setIsEditing(false);
        let finalValue = localValue;
        
        // Validate date
        if (localValue) {
            const date = new Date(localValue);
            const year = date.getFullYear();
            
            if (year >= 2014 && year <= 2030 && !isNaN(date.getTime())) {
                finalValue = localValue;
            } else {
                finalValue = '';
            }
        }
        
        onChange(finalValue);
        setLocalValue(finalValue);
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = e.target.value;
        
        // Validate on change
        if (newValue) {
            const date = new Date(newValue);
            const year = date.getFullYear();
            
            if (year >= 2014 && year <= 2030 && !isNaN(date.getTime())) {
                setLocalValue(newValue);
            }
            // If invalid, don't update localValue
        } else {
            setLocalValue(newValue);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            setIsEditing(false);
            let finalValue = localValue;
            
            if (localValue) {
                const date = new Date(localValue);
                const year = date.getFullYear();
                
                if (year >= 2014 && year <= 2030 && !isNaN(date.getTime())) {
                    finalValue = localValue;
                } else {
                    finalValue = '';
                }
            }
            
            onChange(finalValue);
            setLocalValue(finalValue);
        }
        if (e.key === 'Escape') {
            setLocalValue(value);
            setIsEditing(false);
        }
    };

    return (
        <div className="flex flex-col" ref={containerRef}>
            <label className="text-xs uppercase tracking-wider text-gray-400 mb-1 font-medium">
                {label}
            </label>
            {isEditing ? (
                <input
                    ref={inputRef}
                    type="date"
                    value={localValue}
                    onChange={handleChange}
                    onBlur={handleBlur}
                    onKeyDown={handleKeyDown}
                    className="bg-transparent border-b border-blue-400 focus:outline-none text-gray-800 text-lg py-1 px-0 w-full"
                    placeholder={placeholder}
                    min="2014-01-01"
                    max="2030-12-31"
                    step="1"
                    // Add wheel event for smoother year scrolling
                    onWheel={(e) => {
                        if (inputRef.current) {
                            e.preventDefault();
                            const currentDate = inputRef.current.valueAsDate;
                            if (currentDate) {
                                const newDate = new Date(currentDate);
                                if (e.deltaY < 0) {
                                    newDate.setFullYear(newDate.getFullYear() + 1);
                                } else {
                                    newDate.setFullYear(newDate.getFullYear() - 1);
                                }
                                
                                // Validate year range
                                const year = newDate.getFullYear();
                                if (year >= 2014 && year <= 2030) {
                                    const yearStr = newDate.toISOString().split('T')[0];
                                    setLocalValue(yearStr);
                                    onChange(yearStr);
                                }
                            } else {
                                // If no date selected, set to current valid year
                                const defaultDate = new Date('2024-01-01');
                                setLocalValue(defaultDate.toISOString().split('T')[0]);
                            }
                        }
                    }}
                />
            ) : (
                <div
                    onClick={() => setIsEditing(true)}
                    className={`cursor-text text-lg py-1 border-b border-transparent hover:border-blue-300 transition-all duration-200 group flex items-center justify-between ${
                        value ? 'text-gray-800' : 'text-gray-300'
                    }`}
                >
                    <span className={!value ? 'opacity-50' : 'opacity-100'}>
                        {value || placeholder}
                    </span>
                    <svg 
                        className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity duration-200 text-blue-400" 
                        fill="none" 
                        stroke="currentColor" 
                        viewBox="0 0 24 24"
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                    </svg>
                </div>
            )}
        </div>
    );
}

// EditableTextField Component - for text inputs
function EditableTextField({ 
    value, 
    onChange, 
    placeholder, 
    label,
    type = 'text'
}: { 
    value: string; 
    onChange: (val: string) => void; 
    placeholder: string; 
    label: string;
    type?: string;
}) {
    const [isEditing, setIsEditing] = useState(false);
    const [localValue, setLocalValue] = useState(value);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (isEditing && inputRef.current) {
            inputRef.current.focus();
        }
    }, [isEditing]); // Only depends on isEditing

    const handleBlur = () => {
        setIsEditing(false);
        onChange(localValue);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            setIsEditing(false);
            onChange(localValue);
        }
        if (e.key === 'Escape') {
            setLocalValue(value);
            setIsEditing(false);
        }
    };

    return (
        <div className="flex flex-col">
            <label className="text-xs uppercase tracking-wider text-gray-400 mb-1 font-medium">
                {label}
            </label>
            {isEditing ? (
                <input
                    ref={inputRef}
                    type={type}
                    value={localValue}
                    onChange={(e) => setLocalValue(e.target.value)}
                    onBlur={handleBlur}
                    onKeyDown={handleKeyDown}
                    className="bg-transparent border-b border-blue-400 focus:outline-none text-gray-800 text-lg py-1 px-0 w-full"
                    placeholder={placeholder}
                />
            ) : (
                <div
                    onClick={() => setIsEditing(true)}
                    className={`cursor-text text-lg py-1 border-b border-transparent hover:border-blue-300 transition-all duration-200 group flex items-center justify-between ${
                        value ? 'text-gray-800' : 'text-gray-300'
                    }`}
                >
                    <span className={!value ? 'opacity-50' : 'opacity-100'}>
                        {value || placeholder}
                    </span>
                    <svg 
                        className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity duration-200 text-blue-400" 
                        fill="none" 
                        stroke="currentColor" 
                        viewBox="0 0 24 24"
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                    </svg>
                </div>
            )}
        </div>
    );
}

export default function Header({
    selectedCrime,
    setSelectedCrime,
    selectedNeighborhood,
    setSelectedNeighborhood,
    startDate,
    setStartDate,
    endDate,
    setEndDate,
    onSubmit
}: HeaderProps) {
    const [isTitleHovered, setIsTitleHovered] = useState(false);

    // Wrapper functions to validate dates
    const handleStartDateChange = (date: string) => {
        const validated = validateAndFormatDate(date);
        setStartDate(validated);
    };

    const handleEndDateChange = (date: string) => {
        const validated = validateAndFormatDate(date);
        setEndDate(validated);
    };

    return (
        <header className="relative overflow-visible">
            {/* Gradient background that becomes transparent at the bottom */}
            <div className="absolute inset-0 bg-gradient-to-b from-blue-50 via-white/80 to-transparent pointer-events-none" />
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-100/30 via-transparent to-transparent pointer-events-none" />
            
            <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 pb-12">
                <div className="flex flex-col space-y-10">
                    {/* Full-width Title and Description - all on same line */}
                    <div 
                        className="w-full flex flex-wrap items-center justify-center gap-2 md:gap-4 py-2"
                        onMouseEnter={() => setIsTitleHovered(true)}
                        onMouseLeave={() => setIsTitleHovered(false)}
                    >
                        <h1 className={`
                            text-2xl md:text-3xl font-medium tracking-tight
                            font-['Book_Antiqua',_Palatino,_serif]
                            transition-all duration-300
                            ${isTitleHovered 
                                ? 'bg-gradient-to-r from-gray-600 via-blue-500 to-purple-500 bg-clip-text text-transparent [background-size:200%_auto] [animation:gradient-flow_4s_linear_infinite]' 
                                : 'text-gray-500/70'
                            }
                        `}>
                            Crime detector
                        </h1>
                        <span className="text-gray-400/50 text-xl">•</span>
                        <p className={`
                            text-lg md:text-xl font-normal tracking-wide
                            font-['Book_Antiqua',_Palatino,_serif]
                            transition-all duration-300
                            ${isTitleHovered 
                                ? 'bg-gradient-to-r from-gray-600 via-blue-500 to-purple-500 bg-clip-text text-transparent [background-size:200%_auto] [animation:gradient-flow_4s_linear_infinite]' 
                                : 'text-gray-500/70'
                            }
                        `}>
                            AI-powered crime risk assessment for Toronto neighborhoods
                        </p>
                    </div>

                    {/* Form with enhanced dropdowns */}
                    <form onSubmit={onSubmit} className="space-y-8">
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-end">
                            {/* Crime Type - Enhanced Dropdown */}
                            <EnhancedDropdown
                                value={selectedCrime}
                                onChange={setSelectedCrime}
                                placeholder="Select crime type"
                                label="CRIME TYPE"
                                options={crimeTypes}
                            />

                            {/* Neighborhood - Enhanced Dropdown */}
                            <EnhancedDropdown
                                value={selectedNeighborhood}
                                onChange={setSelectedNeighborhood}
                                placeholder="Search by name or ID..."
                                label="NEIGHBORHOOD"
                                options={neighborhoods}
                            />

                            {/* Start Date - Custom date input with enhanced scrolling */}
                            <DateInput
                                value={startDate}
                                onChange={handleStartDateChange}
                                placeholder="YYYY-MM-DD"
                                label="START DATE"
                            />

                            {/* End Date - Custom date input with enhanced scrolling */}
                            <DateInput
                                value={endDate}
                                onChange={handleEndDateChange}
                                placeholder="YYYY-MM-DD"
                                label="END DATE"
                            />
                        </div>

                        {/* Minimal Text-Based Submit without arrow */}
                        <div className="flex justify-center pt-2">
                            <button
                                type="submit"
                                className="group relative inline-flex items-center gap-2 text-blue-600 font-medium text-lg tracking-wide transition-all duration-300 hover:text-blue-700 focus:outline-none"
                            >
                                <span className="relative">
                                    Predict Crime Risk
                                </span>
                                {/* Subtle glow on hover */}
                                <span className="absolute inset-0 rounded-full blur-md bg-blue-400/0 group-hover:bg-blue-400/20 transition-all duration-300 -z-10" />
                            </button>
                        </div>
                    </form>
                </div>
            </div>

            {/* Global animation keyframes for gradient flow */}
            <style jsx global>{`
                @keyframes gradient-flow {
                    0% { background-position: 0% 50%; }
                    100% { background-position: 200% 50%; }
                }
                
                @keyframes slide-in-from-top-2 {
                    from {
                        opacity: 0;
                        transform: translateY(-8px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                
                .animate-in {
                    animation-duration: 200ms;
                    animation-fill-mode: both;
                }
                
                .slide-in-from-top-2 {
                    animation-name: slide-in-from-top-2;
                }
            `}</style>
        </header>
    );
}