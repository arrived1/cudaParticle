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

/*
#include "particleSystem.h"
#include "emiter.h"

const float a = 10.f;
const float dt = 0.001f;
const float pointSize = 1.f;

const float v = 16.f;
const float radius = 0.02f;




//unsigned step = 0;
ParticleSystem<float2> pSystem(40);
Emiter<ParticleSystem<float2>, float2> emiter(pSystem, make_float2(-5.f, 0.f));

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

*/




// Author: Igor Ševo
// http://www.igorsevo.com/Article.aspx?article=Million+particles+in+CUDA+and+OpenGL
// igor@igorsevo.com
//
// This source is subject to the Attribution Assurance License.
//
// THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY 
// KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.

#define _USE_MATH_DEFINES
#include <iostream>
#include <math.h>
#include <ctime>
#include <gl\glew.h>
#include <gl\glut.h>

const unsigned int window_width = 768;
const unsigned int window_height = 768;

const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

float rnd1[mesh_width*mesh_height];
float rnd2[mesh_width*mesh_height];

int mouse_x, mouse_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

float anim = 0.0;

struct float3
{
    float x, y, z;
};

struct float4
{
    float x, y, z, w;
};

float3 make_float3(float X, float Y, float Z)
{
    float3 temp = {X,Y,Z};
    return temp;
}

float4 make_float4(float X, float Y, float Z, float W)
{
    float4 temp = {X,Y,Z, W};
    return temp;
}

float3 pos[mesh_width][mesh_height];
float3 vel[mesh_width][mesh_height];
float4 col[mesh_width][mesh_height];

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case(27) :
        exit(0);
        break;
    case('a'):
        if (mouse_buttons!=10)
            mouse_buttons=10;
        else
            mouse_buttons=0;
        break;
    }
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_x = x;
    mouse_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_x;
    dy = y - mouse_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01;
    }

    mouse_x = x;
    mouse_y = y;
}

void initialize()
{
    #pragma omp parallel for
    for (int x=0;x<mesh_width;++x)
        for (int y=0;y<mesh_height;++y)
        {
            float u = x / (float) mesh_width + rnd1[y*mesh_width+x];
            float v = y / (float) mesh_height + rnd2[y*mesh_height+x];

            float freq = 2.0f;
            float w = sin(u*freq + anim) * cos(v*freq + anim) * 0.2f;

            col[y][x] = make_float4(0,255,255,1);
            pos[y][x] = make_float3(u, w, v);
            vel[y][x] = make_float3(0.0, 0.0, 0.0);
        }
}

void particles()
{
    const float speed = 0.0005f;
    const float threshold = 0.1f;
    #pragma omp parallel for
    for (int x=0;x<mesh_width;++x)
        for (int y=0;y<mesh_height;++y)
        {
            float u = x / (float) mesh_width;
            float v = y / (float) mesh_height;
            float xX = (mouse_x - (float)mesh_width/2-256)/(float)mesh_width/2;
            float yY = (mouse_y - (float)mesh_height/2-256)/(float)mesh_height/2;
            float dx = -pos[y][x].x + xX;
            float dz = -pos[y][x].z + yY;
            float length = sqrt(dx*dx+dz*dz);
            if (mouse_buttons==10)
            {
                vel[y][x].x=0;
                vel[y][x].z=0;
                dx = -pos[y][x].x + u;
                dz = -pos[y][x].z + v;
                length = sqrt(dx*dx+dz*dz);
                pos[y][x].x+=dx/length*speed*10;
                pos[y][x].z+=dz/length*speed*10;
            }
            else if (!(mouse_buttons & 4) && !(mouse_buttons & 6))
            {
                float norX = dx/length*speed;
                float norZ = dz/length*speed;
                vel[y][x].x+=norX;
                vel[y][x].z+=norZ;
                dx = vel[y][x].x;
                dz = vel[y][x].z;
                float velocity = sqrt(dx*dx+dz*dz);
                if (velocity>threshold)
                {
                    vel[y][x].x=dx/velocity*threshold;
                    vel[y][x].z=dz/velocity*threshold;
                }
                float green = (int)(255/(velocity*51))/255.0f;
                if (green>=1.0f)
                    green=1.0f;
                col[y][x] = make_float4(128/length/255.0f,green,1.0,0.1);
                if (pos[y][x].x<-5.0f && vel[y][x].x<0.0)
                    vel[y][x].x=-vel[y][x].x;
                if (pos[y][x].x>5.0f && vel[y][x].x>0.0)
                    vel[y][x].x=-vel[y][x].x;
                pos[y][x].x+=vel[y][x].x;
                pos[y][x].z+=vel[y][x].z;
            }
            else if (!(mouse_buttons & 4))
            {
                vel[y][x].x=0;
                vel[y][x].z=0;
                pos[y][x].x+=dx/length*speed*10;
                pos[y][x].z+=dz/length*speed*10;
                col[y][x] = make_float4(1.0f/length,1.0f/length, 1.0f, 10);
            }
            float freq = 2.0f;
            float w = sin(u*freq + anim) * cos(v*freq + anim) * 0.2f;
            pos[y][x].y=w;
        }
}

static void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    particles();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(90.0, 1.0, 0.0, 0.0);

    glBegin(GL_POINTS);
        for (int x=0;x<mesh_width;++x)
            for (int y=0;y<mesh_height;++y)
            {
                glColor4f(col[y][x].x, col[y][x].y, col[y][x].z, col[y][x].z);
                glVertex3f(pos[y][x].x, pos[y][x].y, pos[y][x].z);
            }
    glEnd();

    glutSwapBuffers();
    glutPostRedisplay();

    anim += 0.01;
}

void initGL(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Million particles");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);

    glewInit();
    if (! glewIsSupported("GL_VERSION_2_0 ")) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        exit(0);
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, window_width, window_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
}

int main(int argc, char** argv)
{
    initGL(argc, argv);

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    for (int i=0;i<mesh_height*mesh_width;++i)
        rnd1[i]=(rand()%100-100)/2000.0f;
    for (int i=0;i<mesh_height*mesh_width;++i)
        rnd2[i]=(rand()%100-100)/2000.0f;

    initialize();

    glutMainLoop();

    return 0;
}