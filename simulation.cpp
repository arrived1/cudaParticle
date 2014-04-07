#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>

#include <vector>
#include <utility>
#include <cmath>
#include <string>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <cuda.h>


#include "particleSystem.h"
#include "emiter.h"

const float a = 10.f;
const float dt = 0.001f;
const float pointSize = 1.f;

const float v = 16.f;
const float radius = 0.02f;




//unsigned step = 0;
ParticleSystem<float2> pSystem(40);
float2 pos = make_float2(-5.f, 0.f);
Emiter<ParticleSystem<float2>, float2> emiter(pSystem, pos);

bool collision(float2 p1, float2 p2);
void checkState(unsigned i);

void renderDots() {
    glPointSize(pointSize);
    glBegin(GL_POINTS);
    
    float3 color;
    for (unsigned i = 0; i < pSystem.getParticleAmount(); i++) {
        color = pSystem.getParticleColor(i);
        glColor3f(color.x, color.y, color.z);
        glVertex3f(pSystem.getParticlePos(i).x, pSystem.getParticlePos(i).y, 0.0f);
    }
    glEnd();
}

void renderEmiter() {
    glPointSize(10.f);
    float2 pos = emiter.getPosition();

    glBegin(GL_POINTS);
        glColor3f(1.f, 1.f, 1.f);    
        glVertex3f(pos.x, pos.y, 0.0f);
    glEnd();
}

void renderLines() {
    glBegin(GL_LINES);
    glColor3f(0.f, 1.f, 0.f);
        glVertex2f(-4.f, 10.f);
        glVertex2f(-4.f, -10.f);

        glVertex2f(4.8f, 10.f);
        glVertex2f(4.8f, -10.f);
    glEnd();
}

void display(void) {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Clear the background of our window to red  
    glClear(GL_COLOR_BUFFER_BIT); //Clear the colour buffer (more buffers later on)  
    glLoadIdentity(); // Load the Identity Matrix to reset our drawing locations  

    glTranslatef(0.0f, 0.0f, -(a+2)); // Push eveything 5 units back into the scene, otherwise we won't see the primitive  

    //renderPrimitive(); // Render the primitive
    
    renderDots();
    renderEmiter();
    renderLines();
    glFlush(); // Flush the OpenGL buffers to the window  
}

void timer_function(int value) {
    for (unsigned i = 0; i < pSystem.getParticleAmount(); i++)
        emiter.emit(i);

    std::cout << "Startuje synulacje" << std::endl;
    for (unsigned i = 0; i < pSystem.getParticleAmount(); i++) 
        for (unsigned j = 0; j < pSystem.getParticleAmount(); j++) {
            
            if(i == j)
                continue;

            checkState(i);            

            if (pSystem.getParticleState(i) == inBox) {
                if (collision(pSystem.getParticlePos(i), pSystem.getParticlePos(j))) {
                    std::cout << "kolizja!!! " << i << " z " << j << std::endl;
                    float2 vel = pSystem.getParticleVel(i);
                    vel.x *= -1;
                    vel.y *= -1;
                    pSystem.setParticleVel(i, vel);

                    pSystem.getNextParticlePos(i) = pSystem.getPrevParticlePos(i);
                }
            }
        }
    


    for (unsigned i = 0; i < pSystem.getParticleAmount(); i++) {
        pSystem.prepareMove(i, dt);
        pSystem.move(i);
    }

    pSystem.incrementStep();


    glutPostRedisplay();
    glutTimerFunc(200, timer_function, 0);
}

bool collision(float2 p1, float2 p2) {
    float d_kwadrat = (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y);
    float r_kwadrat = (2 * radius) * (2 * radius);

    if (d_kwadrat <= r_kwadrat) 
        return true;

    return false;
}

void checkState(unsigned i) {
    int x = pSystem.getParticlePos(i).x;
    
    if (x > -4.f)
        pSystem.setParticleState(i, inBox);
    
    if (x > 4.8f) {
        emiter.backToEmiter(i);
    }   
}

void reshape(int width, int height) {
    
    glViewport(0, 0, (GLsizei)width, (GLsizei)height); // Set our viewport to the size of our window  
    glMatrixMode(GL_PROJECTION); // Switch to the projection matrix so that we can manipulate how our scene is viewed  
    glLoadIdentity(); // Reset the projection matrix to the identity matrix so that we don't get any artifacts (cleaning up)  
    gluPerspective(60, (GLfloat)width / (GLfloat)height, 1.0, 100.0); // Set the Field of view angle (in degrees), the aspect ratio of our window, and the new and far planes  
    glMatrixMode(GL_MODELVIEW); // Switch back to the model view matrix, so that we can start drawing shapes correctly  
}


int main(int argc, char **argv) {
    //initParticle();

    //std::cout << particles << std::endl;
    glutInit(&argc, argv); // Initialize GLUT  
    glutInitDisplayMode(GLUT_SINGLE); // Set up a basic display buffer (only single buffered for now)  
    glutInitWindowSize(500, 500); // Set the width and height of the window  
    glutInitWindowPosition(500, 100); // Set the position of the window  
    glutCreateWindow("Your first OpenGL Window"); // Set the title for the window  

    glutDisplayFunc(display); // Tell GLUT to use the method "display" for rendering  
    glutTimerFunc(1, timer_function, 0);

    glutReshapeFunc(reshape); // Tell GLUT to use the method "reshape" for reshaping  

    glutMainLoop(); // Enter GLUT's main loop  
}

