var i = 0;
var speed = 40;

function addexample(){
    clicked=true
    $(".chatbox").append(`<li class="chat outgoing">
                             <div class="user-que">Request to access Dashboard ?</div>
                         </li>`)
    $(".chatbox").append(`<li class="chat incoming" id="example-ans" style="">
                               <span class="material-symbols-outlined">smart_toy</span>
                               <div>
                                  <div class="bot-response"></div>
                                </div>
                         </li>`)                  
    
    let txt2=`1 . Click on the 'Raise new request' button located on the top right of the page.\n2 . Enter the OHR and click anywhere outside the field box.\n3 . The email will auto-populate once the OHR is entered.\n4 . The 'Read' option is set to Read-only by default for security access.\n5 . Choose the type of data access you need for the dashboard (Financial, Non-Financial, Sales).\n6 . Based on the type selected, fill in the corresponding Security Types and Security Values.\n7 . Ensure all Security Types and their values are filled in. If you need access to all values under a specific type, choose 'ALL'.\n8 . For Financial information, select Actuals/Outlook/OP Plan.\n9 . For Non-Financial information, select Headcount (Genpact and contractors) and Recruitment.\n10 . For Sales information, select Inflow, Booking, and Pipeline details.\nSimilarly You can ask the question in the below provided chat input.\n`              
    typeWriter(txt2.replace(/\n/g,"<br>"),typingSpeed = 40)

    $('.example-btn').prop('disabled', true).css({
        'background-color': '#e6878d',
        'color': 'black'
    });
}

function typeWriter(text,typingSpeed = 40) {
    console.log(text);
    let ele=document.getElementsByClassName("bot-response")
    let previoustext=ele[ele.length-1].innerHTML
    let displayedText = "";
    let i = 0;
    function type() {
      if (i < text.length) {
       // console.log('i',i);
        displayedText =previoustext + text.substring(0, i + 1);
        ele[ele.length-1].innerHTML = displayedText;
        i++;
        if(i== text.length){
            $('.like-btn-parent').last().css('display', 'block');
        }
        setTimeout(type, typingSpeed);
      }
    }
    type();
  }

function like(){
    var lastOutgoingText = $('.u-question:last').text().trim();
    var lastBotResponseText = $('.bot-response:last').text().trim();
    var formData = new FormData();
    formData.append('usermassage', lastOutgoingText);
    formData.append('botresponse', lastBotResponseText);
    formData.append('datetime', realtime_datetime());
    $.ajax({
        url: '/likeresponse',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            console.log(response);
            $('#res-alert').fadeIn().delay(3000).fadeOut();
            $('.like-btn-parent').css({
                'opacity': '0.6',
                'pointer-events': 'none'
            });
        },
        error: function(xhr, status, error) {
            console.error(error);
        }
    });
}
function dislike(){
    var lastOutgoingText = $('.u-question:last').text().trim();
    var lastBotResponseText = $('.bot-response:last').text().trim();
        var formData = new FormData();
        formData.append('usermassage', lastOutgoingText);
        formData.append('botresponse', lastBotResponseText);
        formData.append('datetime', realtime_datetime());
        $.ajax({
            url: '/dislikeresponse',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                console.log(response);
                $('#res-alert').fadeIn().delay(3000).fadeOut();
                $('.like-btn-parent').css({
                    'opacity': '0.6',
                    'pointer-events': 'none'
                });
            },
            error: function(xhr, status, error) {
                console.error(error);
            }
        });
}

function realtime_datetime (){
    var currentDatetime = new Date();
    var formattedDatetime = currentDatetime.toISOString();
    return formattedDatetime 

}



