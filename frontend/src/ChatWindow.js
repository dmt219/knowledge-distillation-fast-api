import React, { useState } from "react";
import axios from "axios";
import "./styles.css";

const ChatWindow = () => {
  const [chatHistory, setChatHistory] = useState([]);

  const handleUserMessage = async (messageText) => {
    const newUserMessage = { text: messageText, sender: "user" };
    setChatHistory((chatHistory) => [...chatHistory, newUserMessage]);

    try {
      const response = await axios.post("http://localhost:8000/generate", {
        message: messageText,
      });
      const botResponse = { text: response.data.message, sender: "bot" };
      setChatHistory((chatHistory) => [...chatHistory, botResponse]);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="chat-window">
      <MessageList chatHistory={chatHistory} />
      <MessageInput onMessageSubmit={handleUserMessage} />
    </div>
  );
};

const MessageList = ({ chatHistory }) => {
  return (
    <ul className="message-list">
      {chatHistory.map((message, index) => (
        <li key={index} className={`message ${message.sender}`}>
          {message.text}
        </li>
      ))}
    </ul>
  );
};

const MessageInput = ({ onMessageSubmit }) => {
  const [inputMessage, setInputMessage] = useState("");

  const handleSubmit = () => {
    if (inputMessage.trim() !== "") {
      onMessageSubmit(inputMessage);
      setInputMessage("");
    }
  };

  return (
    <div className="message-input">
      <input
        type="text"
        value={inputMessage}
        onChange={(e) => setInputMessage(e.target.value)}
        placeholder="Type your message..."
      />
      <button onClick={handleSubmit}>Send</button>
    </div>
  );
};

export default ChatWindow;
