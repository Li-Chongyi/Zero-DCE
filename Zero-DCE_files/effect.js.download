function imageZoom(img_container_ID, fix) {
  'use strict'
  let container = document.getElementById(img_container_ID)
  let leftView = container.getElementsByClassName('leftView')[0]
  let mask = container.getElementsByClassName('mask')[0]
  let rightView = container.getElementsByClassName('rightView')[0]
  let bigImg = container.getElementsByClassName('big')[0]
  leftView.addEventListener('mouseover', function() {
    mask.style.display = 'block'
    rightView.style.display = 'block'
  }, false)
  leftView.addEventListener('mouseout', function() {
    mask.style.display = 'none'
    rightView.style.display = 'none'
  })
  // console.log(leftView);
  leftView.addEventListener('mousemove', function(evt) {
    evt = evt || window.event
    let currentMouseX = evt.pageX
    let currentMouseY = evt.pageY
    let offsetLeft = container.offsetLeft + leftView.offsetLeft
    let offsetTop = container.offsetTop + leftView.offsetTop
    let maskWidth = mask.offsetWidth
    let maskHeight = mask.offsetHeight
    let zoomMaskX = currentMouseX - offsetLeft - maskWidth / 2
    let zoomMaskY = currentMouseY - offsetTop - maskHeight / 2
    if (zoomMaskX <= -maskWidth / 2) {
      zoomMaskX = -maskWidth / 2
    }
    if (zoomMaskY <= -maskWidth / 2) {
      zoomMaskY = -maskWidth / 2
    }
    let maxScopeX = leftView.offsetWidth - maskWidth / 2
    if (zoomMaskX >= maxScopeX) {
      zoomMaskX = maxScopeX
    }
    let maxkScopeY = leftView.offsetHeight - maskHeight / 2
    if (zoomMaskY >= maxkScopeY) {
      zoomMaskY = maxkScopeY
    }
    mask.style.left = zoomMaskX + leftView.offsetLeft + 'px'
    mask.style.top = zoomMaskY + leftView.offsetTop + 'px'
    // rightView position
    if (fix > 0) {
      rightView.style.left = container.offsetWidth - 400 * 1.1 + 'px'
      rightView.style.top = container.offsetHeight + 10 + 'px'
    } else {
      rightView.style.left = currentMouseX + 30 + 'px'
      rightView.style.top = currentMouseY + 30 + 'px'
    }
    let zommProportion = bigImg.offsetWidth / leftView.offsetWidth
    bigImg.style.left = -zommProportion * (zoomMaskX + maskWidth / 2) + rightView.offsetWidth / 2 + 'px'
    bigImg.style.top = -zommProportion * (zoomMaskY + maskHeight / 2) + rightView.offsetHeight / 2 + 'px'

  }, false)
}
// light-box
function openModal() {
  document.getElementById('myModal').style.display = "block";
}

function closeModal() {
  document.getElementById('myModal').style.display = "none";
}

function plusSlides(n) {
  showSlides(slideIndex += n);
}

function currentSlide(n) {
  showSlides(slideIndex = n);
}

function showSlides(n) {
  var i;
  var slides = document.getElementsByClassName("mySlides");
  var dots = document.getElementsByClassName("dot");

  if (n > slides.length) {
    slideIndex = 1
  }
  if (n < 1) {
    slideIndex = slides.length
  }
  for (i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
  }
  for (i = 0; i < dots.length; i++) {
    dots[i].className = dots[i].className.replace(" active", "");
  }
  slides[slideIndex - 1].style.display = "block";
  dots[slideIndex - 1].className += " active";
}