'use client'
import { useState, useEffect, useRef } from 'react';

export default function Home() {

    //
    const [darkTheme, setDarkTheme] = useState(true);
    const [userInput, setUserInput] = useState('');
    const [conversations, setConversations] = useState([]);
    const [conversationHistory, setConversationHistory] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    //
    const messagesEndRef = useRef(null); // Create a ref

    //
    useEffect(() => {
        document.documentElement.classList.toggle('dark', darkTheme);
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [darkTheme, conversations]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const userText = userInput.trim();
        if (!userText || isLoading) return;

        setIsLoading(true);
        setConversations((prev) => [...prev, { text: userText, sender: 'user' }]);

        const messageWithMarkers = `##@${userText}@##`;
        const fullInputForServer = conversationHistory + `Instruction:\n${messageWithMarkers}\n\n`;
        //const updatedHistory = conversationHistory + `Instruction:\n${userText}\n\n`;
        setUserInput('');
        console.log(`fullInputForServer: ${fullInputForServer}`)

        const res = await fetch('http://localhost:8000/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: fullInputForServer }),
        });
        const data = await res.json();

        setConversations((prev) => [...prev, { text: data.response, sender: 'bot' }]);
        //setConversationHistory(prev => prev + `Response:\n${data.response}\n\n`);
        setConversationHistory(prev => prev + `Instruction:\n${userText}\n\nResponse:\n${data.response}\n\n`);
        setIsLoading(false);
    };

    return (
        <div className={`flex flex-col h-screen ${darkTheme ? 'dark' : ''}`}>
            <nav className="bg-pink-500 text-white p-4 flex justify-center items-center dark:bg-black">
                <h1 className="font-bold text-center">NLP Project Chatbot</h1>
                <div className="absolute right-4">
                    <button
                        onClick={() => setDarkTheme(!darkTheme)}
                        className="text-white font-bold py-2 px-4 rounded"
                        style={{ backgroundColor: darkTheme ? 'white' : 'pink', color: darkTheme ? 'black' : 'white' }}
                    >
                        {darkTheme ? 'Light' : 'Dark'} Mode
                    </button>
                </div>
            </nav>
            <div className="flex-grow p-4 overflow-y-scroll bg-white dark:bg-black">
                {conversations.map((msg, idx) => (
                    <div key={idx} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`rounded-lg p-2 my-2 ${msg.sender === 'user' ? 'bg-pink-500' : 'bg-pink-300'} dark:bg-gray-700`}>
                            <p className="text-white dark:text-white">{msg.text}</p>
                        </div>
                    </div>
                ))}
                {/* Empty div to act as the scroll target */}
                <div ref={messagesEndRef} />
            </div>
            <form onSubmit={handleSubmit} className="flex items-center border-t-2 border-gray-200 p-4 dark:border-gray-700 bg-white dark:bg-black">
                <input
                    type="text"
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    placeholder="Type your message here..."
                    disabled={isLoading} // Disable input when loading
                    className="w-full mr-4 rounded-md p-2 border-gray-300 focus:border-pink-500 focus:ring focus:ring-pink-500 focus:ring-opacity-50 dark:bg-gray-900 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white"
                />
                <button
                    type="submit"
                    className="bg-pink-500 hover:bg-pink-700 text-white font-bold py-2 px-4 rounded dark:bg-white dark:text-black"
                    disabled={isLoading || !userInput.trim()} // Disable button when loading or input is empty
                >
                    {isLoading ? 'Wait' : 'Send'}
                </button>
            </form>
        </div>
    );
}