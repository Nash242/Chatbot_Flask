var i = 0;
var speed = 50;

function addexample(){
    clicked=true
    $(".chatbox").append(`<li class="chat outgoing">
                             <div class="user-que">Request to access Dashboard ?</div>
                         </li>`)
    $(".chatbox").append(`<li class="chat incoming" id="example-ans" style="">
                               <span class="material-symbols-outlined">smart_toy</span>
                                <div>
                                1 . Click on the 'Raise new request' button located on the top right of the page. <br>
                                2 . Enter the OHR and click anywhere outside the field box.<br>
                                3 . The email will auto-populate once the OHR is entered.<br>
                                4 . The 'Read' option is set to Read-only by default for security access.<br>
                                5 . Choose the type of data access you need for the dashboard (Financial, Non-Financial, Sales).<br>
                                6 . Based on the type selected, fill in the corresponding Security Types and Security Values.<br>
                                7 . Ensure all Security Types and their values are filled in. If you need access to all values under a specific type, choose 'ALL'.<br>
                                8 . For Financial information, select Actuals/Outlook/OP Plan.<br>
                                9 . For Non-Financial information, select Headcount (Genpact and contractors) and Recruitment.<br>
                                10 . For Sales information, select Inflow, Booking, and Pipeline details.<br>
                                Similarly You can ask the question in the below provided chat input. </div>
                         </li>`)                  
    

    $('.example-btn').prop('disabled', true).css({
        'background-color': '#e6878d',
        'color': 'black'
    });
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


// function typeWriter(txt) {
//     if (i < txt.length) {
//         let ele=document.getElementsByClassName("incoming")
//         ele[ele.length-1].innerHTML += txt.charAt(i)+txt.charAt(i+1)+txt.charAt(i+2);
//         i+=3;
//         setTimeout(typeWriter(txt), speed);
//     }
// }
function typeWriter(txt) {
    if (i < txt.length) {
        let ele = document.getElementsByClassName("incoming");
        ele[ele.length - 1].innerHTML += txt.charAt(i) + txt.charAt(i + 1) + txt.charAt(i + 2);
        if (txt.charAt(i) === '\n') {
            ele[ele.length - 1].appendChild(document.createElement('br'));
        }
        i += 3;
        setTimeout(function() {
            typeWriter(txt);
        }, speed);
    }
}