const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const refreshbtn = document.querySelector(".refresh-btn");

var clicked=false;
let userMessage = null; // Variable to store user's message
const API_KEY = "PASTE-YOUR-API-KEY"; // Paste your API key here
const inputInitHeight = chatInput.scrollHeight;

// const createChatLi = (message, className) => {
//     const chatLi = document.createElement("li");
//     chatLi.classList.add("chat", `${className}`);
//     let chatContent = className === "outgoing" ? `<div class='u-question'></div>` : `<span class="material-symbols-outlined">smart_toy</span><div></div>`;
//     chatLi.innerHTML = chatContent;
//     chatLi.querySelector("div").textContent = message;
//     return chatLi; // 
// }

const generateResponse = (usermsg) => {
    let html_msg=`<li class="chat outgoing"><div class="u-question">${usermsg}</div></li>`;
    $(".chatbox").append(html_msg)
    let loader =`<li class="chat incoming"><span class="material-symbols-outlined">smart_toy</span>
                    <div>
                        <div class="bot-response"><div class="bot-loader"></div></div>
                    </div>
                </li>`
    $(".chatbox").append(loader)
    const API_URL = "/getanswer";
    const formData = new FormData();
    formData.append("question", userMessage);
    $.ajax({
        url: API_URL,
        type: "POST",
        data: formData,
        processData: false,  
        contentType: false  
    }).done(function(data) {
        console.log(data);
        $('.incoming:last').remove();
        var res=''
        if (!data.message.startsWith("answer:")) {
            res = data.message.trim() //.replace(/\n/g, " <br> ");
        } else { 
            res=data.message.trim()
        }
        if(data.status == 'success'){
            $(".chatbox").append(`<li class="chat incoming"><span class="material-symbols-outlined">smart_toy</span>
                                        <div>
                                            <div class="bot-response"></div>
                                            <div class="like-btn-parent" style="display:none">
                                            <div style="display: flex;gap: 15px;" id="svg-parent ">
                                            <svg onclick="like()" class="like-btn" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path fill="#FFD43B" d="M313.4 32.9c26 5.2 42.9 30.5 37.7 56.5l-2.3 11.4c-5.3 26.7-15.1 52.1-28.8 75.2H464c26.5 0 48 21.5 48 48c0 18.5-10.5 34.6-25.9 42.6C497 275.4 504 288.9 504 304c0 23.4-16.8 42.9-38.9 47.1c4.4 7.3 6.9 15.8 6.9 24.9c0 21.3-13.9 39.4-33.1 45.6c.7 3.3 1.1 6.8 1.1 10.4c0 26.5-21.5 48-48 48H294.5c-19 0-37.5-5.6-53.3-16.1l-38.5-25.7C176 420.4 160 390.4 160 358.3V320 272 247.1c0-29.2 13.3-56.7 36-75l7.4-5.9c26.5-21.2 44.6-51 51.2-84.2l2.3-11.4c5.2-26 30.5-42.9 56.5-37.7zM32 192H96c17.7 0 32 14.3 32 32V448c0 17.7-14.3 32-32 32H32c-17.7 0-32-14.3-32-32V224c0-17.7 14.3-32 32-32z"></path></svg>
                                            <svg onclick="dislike()" style="transform: rotateY(180deg);" class="like-btn" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path fill="#FFD43B" d="M313.4 479.1c26-5.2 42.9-30.5 37.7-56.5l-2.3-11.4c-5.3-26.7-15.1-52.1-28.8-75.2H464c26.5 0 48-21.5 48-48c0-18.5-10.5-34.6-25.9-42.6C497 236.6 504 223.1 504 208c0-23.4-16.8-42.9-38.9-47.1c4.4-7.3 6.9-15.8 6.9-24.9c0-21.3-13.9-39.4-33.1-45.6c.7-3.3 1.1-6.8 1.1-10.4c0-26.5-21.5-48-48-48H294.5c-19 0-37.5 5.6-53.3 16.1L202.7 73.8C176 91.6 160 121.6 160 153.7V192v48 24.9c0 29.2 13.3 56.7 36 75l7.4 5.9c26.5 21.2 44.6 51 51.2 84.2l2.3 11.4c5.2 26 30.5 42.9 56.5 37.7zM32 384H96c17.7 0 32-14.3 32-32V128c0-17.7-14.3-32-32-32H32C14.3 96 0 110.3 0 128V352c0 17.7 14.3 32 32 32z"></path></svg>
                                            </div>
                                            <div id="res-alert" style="font-size: 12px;color: #ff555f;display:none;">
                                                 Response Saved...!
                                            </div>
                                            </div>
                                        </div>
                                    </li>`)   
        }else{
            $(".chatbox").append(`<li class="chat incoming"><span class="material-symbols-outlined">smart_toy</span>
                                        <div>
                                            <div class="bot-response error" style="padding: 4px;border-radius: 0px 6px 6px 6px;"></div>
                                            <div class="like-btn-parent" style="display:none">
                                            <div style="display: flex;gap: 15px;" id="svg-parent ">
                                            <svg onclick="like()" class="like-btn" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path fill="#FFD43B" d="M313.4 32.9c26 5.2 42.9 30.5 37.7 56.5l-2.3 11.4c-5.3 26.7-15.1 52.1-28.8 75.2H464c26.5 0 48 21.5 48 48c0 18.5-10.5 34.6-25.9 42.6C497 275.4 504 288.9 504 304c0 23.4-16.8 42.9-38.9 47.1c4.4 7.3 6.9 15.8 6.9 24.9c0 21.3-13.9 39.4-33.1 45.6c.7 3.3 1.1 6.8 1.1 10.4c0 26.5-21.5 48-48 48H294.5c-19 0-37.5-5.6-53.3-16.1l-38.5-25.7C176 420.4 160 390.4 160 358.3V320 272 247.1c0-29.2 13.3-56.7 36-75l7.4-5.9c26.5-21.2 44.6-51 51.2-84.2l2.3-11.4c5.2-26 30.5-42.9 56.5-37.7zM32 192H96c17.7 0 32 14.3 32 32V448c0 17.7-14.3 32-32 32H32c-17.7 0-32-14.3-32-32V224c0-17.7 14.3-32 32-32z"></path></svg>
                                            <svg onclick="dislike()" style="transform: rotateY(180deg);" class="like-btn" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path fill="#FFD43B" d="M313.4 479.1c26-5.2 42.9-30.5 37.7-56.5l-2.3-11.4c-5.3-26.7-15.1-52.1-28.8-75.2H464c26.5 0 48-21.5 48-48c0-18.5-10.5-34.6-25.9-42.6C497 236.6 504 223.1 504 208c0-23.4-16.8-42.9-38.9-47.1c4.4-7.3 6.9-15.8 6.9-24.9c0-21.3-13.9-39.4-33.1-45.6c.7-3.3 1.1-6.8 1.1-10.4c0-26.5-21.5-48-48-48H294.5c-19 0-37.5 5.6-53.3 16.1L202.7 73.8C176 91.6 160 121.6 160 153.7V192v48 24.9c0 29.2 13.3 56.7 36 75l7.4 5.9c26.5 21.2 44.6 51 51.2 84.2l2.3 11.4c5.2 26 30.5 42.9 56.5 37.7zM32 384H96c17.7 0 32-14.3 32-32V128c0-17.7-14.3-32-32-32H32C14.3 96 0 110.3 0 128V352c0 17.7 14.3 32 32 32z"></path></svg>
                                            </div>
                                            <div id="res-alert" style="font-size: 12px;color: #ff555f;display:none;">
                                                 Response Saved...!
                                            </div>
                                            </div>
                                        </div>
                                    </li>`)    
        }
        res=res.replace(/\n/g,"<br>")
            console.log(res);
        typeWriter(res,typingSpeed = 40)

    }).fail(function() {
        // messageElement.classList.add("error");
        // messageElement.textContent = "Oops! Something went wrong. Please try again.";
    }).always(function() {
        chatbox.scrollTo(0, chatbox.scrollHeight);
    });

}

const handleChat = () => {
    $('.like-btn-parent').remove();
    if (clicked){
        $('.example-div').remove();
        $('.chatbox li').eq(0).remove(); 
        $('.chatbox li').eq(0).remove(); 
        clicked=false
    }else{
        if ($('.example-div').length > 0) {
            $('.example-div').remove();
        }
    }
    userMessage = chatInput.value.trim(); 
    if(!userMessage) return;
    // Clear the input textarea and set its height to default
    chatInput.value = "";
    chatInput.style.height = `${inputInitHeight}px`;
    chatbox.scrollTo(0, chatbox.scrollHeight);
    // setTimeout(() => {
        // Display "Thinking..." message while waiting for the response
        // const incomingChatLi = createChatLi("Thinking...", "incoming");
        // chatbox.appendChild(incomingChatLi);
        // chatbox.scrollTo(0, chatbox.scrollHeight);
        generateResponse(userMessage);
    // }, 600);
}

chatInput.addEventListener("input", () => {
    chatInput.style.height = `${inputInitHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {
    if(e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
        e.preventDefault();
        handleChat();
    }
});

const refreshchat = () => {
    if ($('.example-div').length > 0) {
        $('.example-btn').prop('disabled', false).css({'background-color':'#ff555f','color':'white'});
        $(".chatbox").html('')  
    } else {
        $(".chatbox").html('')
        let htmlstring=`<div class="example-div">
                            <span>
                            click here for demo Q&A
                            </span>
                            <button class="example-btn" onclick="addexample()">Example</button>
                        </div>`
                    const header = document.querySelector('.chatbot header');
                    header.insertAdjacentHTML('afterend', htmlstring);
    }
    console.log('refresh click');
    
}

sendChatBtn.addEventListener("click", handleChat);
closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));
refreshbtn.addEventListener("click", refreshchat);



