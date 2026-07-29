#pragma once
// Minimal OpenGL 3.3 function loader for the parts we use (shaders, buffers,
// VAOs, FBOs). GL 1.1 entry points come from opengl32.lib directly.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>

typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;

#define GL_ARRAY_BUFFER                   0x8892
#define GL_STATIC_DRAW                    0x88E4
#define GL_DYNAMIC_DRAW                   0x88E8
#define GL_FRAGMENT_SHADER                0x8B30
#define GL_VERTEX_SHADER                  0x8B31
#define GL_COMPILE_STATUS                 0x8B81
#define GL_LINK_STATUS                    0x8B82
#define GL_INFO_LOG_LENGTH                0x8B84
#define GL_FRAMEBUFFER                    0x8D40
#define GL_RENDERBUFFER                   0x8D41
#define GL_COLOR_ATTACHMENT0              0x8CE0
#define GL_DEPTH_ATTACHMENT               0x8D00
#define GL_DEPTH_COMPONENT24              0x81A6
#define GL_FRAMEBUFFER_COMPLETE           0x8CD5
#define GL_PROGRAM_POINT_SIZE             0x8642
#define GL_POINT_SPRITE                   0x8861
#define GL_CLAMP_TO_EDGE                  0x812F
#define GL_MULTISAMPLE                    0x809D

typedef GLuint (APIENTRY* PFN_glCreateShader)(GLenum);
typedef void   (APIENTRY* PFN_glShaderSource)(GLuint, GLsizei, const GLchar* const*, const GLint*);
typedef void   (APIENTRY* PFN_glCompileShader)(GLuint);
typedef void   (APIENTRY* PFN_glGetShaderiv)(GLuint, GLenum, GLint*);
typedef void   (APIENTRY* PFN_glGetShaderInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*);
typedef GLuint (APIENTRY* PFN_glCreateProgram)(void);
typedef void   (APIENTRY* PFN_glAttachShader)(GLuint, GLuint);
typedef void   (APIENTRY* PFN_glLinkProgram)(GLuint);
typedef void   (APIENTRY* PFN_glGetProgramiv)(GLuint, GLenum, GLint*);
typedef void   (APIENTRY* PFN_glGetProgramInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*);
typedef void   (APIENTRY* PFN_glUseProgram)(GLuint);
typedef void   (APIENTRY* PFN_glDeleteShader)(GLuint);
typedef void   (APIENTRY* PFN_glDeleteProgram)(GLuint);
typedef GLint  (APIENTRY* PFN_glGetUniformLocation)(GLuint, const GLchar*);
typedef void   (APIENTRY* PFN_glUniformMatrix4fv)(GLint, GLsizei, GLboolean, const GLfloat*);
typedef void   (APIENTRY* PFN_glUniform1f)(GLint, GLfloat);
typedef void   (APIENTRY* PFN_glUniform1i)(GLint, GLint);
typedef void   (APIENTRY* PFN_glGenBuffers)(GLsizei, GLuint*);
typedef void   (APIENTRY* PFN_glBindBuffer)(GLenum, GLuint);
typedef void   (APIENTRY* PFN_glBufferData)(GLenum, GLsizeiptr, const void*, GLenum);
typedef void   (APIENTRY* PFN_glDeleteBuffers)(GLsizei, const GLuint*);
typedef void   (APIENTRY* PFN_glGenVertexArrays)(GLsizei, GLuint*);
typedef void   (APIENTRY* PFN_glBindVertexArray)(GLuint);
typedef void   (APIENTRY* PFN_glDeleteVertexArrays)(GLsizei, const GLuint*);
typedef void   (APIENTRY* PFN_glEnableVertexAttribArray)(GLuint);
typedef void   (APIENTRY* PFN_glVertexAttribPointer)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*);
typedef void   (APIENTRY* PFN_glGenFramebuffers)(GLsizei, GLuint*);
typedef void   (APIENTRY* PFN_glBindFramebuffer)(GLenum, GLuint);
typedef void   (APIENTRY* PFN_glFramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint);
typedef GLenum (APIENTRY* PFN_glCheckFramebufferStatus)(GLenum);
typedef void   (APIENTRY* PFN_glDeleteFramebuffers)(GLsizei, const GLuint*);
typedef void   (APIENTRY* PFN_glGenRenderbuffers)(GLsizei, GLuint*);
typedef void   (APIENTRY* PFN_glBindRenderbuffer)(GLenum, GLuint);
typedef void   (APIENTRY* PFN_glRenderbufferStorage)(GLenum, GLenum, GLsizei, GLsizei);
typedef void   (APIENTRY* PFN_glFramebufferRenderbuffer)(GLenum, GLenum, GLenum, GLuint);
typedef void   (APIENTRY* PFN_glDeleteRenderbuffers)(GLsizei, const GLuint*);

#define PRT_GL_FUNCS(X) \
    X(glCreateShader) X(glShaderSource) X(glCompileShader) X(glGetShaderiv) \
    X(glGetShaderInfoLog) X(glCreateProgram) X(glAttachShader) X(glLinkProgram) \
    X(glGetProgramiv) X(glGetProgramInfoLog) X(glUseProgram) X(glDeleteShader) \
    X(glDeleteProgram) X(glGetUniformLocation) X(glUniformMatrix4fv) X(glUniform1f) \
    X(glUniform1i) X(glGenBuffers) X(glBindBuffer) X(glBufferData) X(glDeleteBuffers) \
    X(glGenVertexArrays) X(glBindVertexArray) X(glDeleteVertexArrays) \
    X(glEnableVertexAttribArray) X(glVertexAttribPointer) X(glGenFramebuffers) \
    X(glBindFramebuffer) X(glFramebufferTexture2D) X(glCheckFramebufferStatus) \
    X(glDeleteFramebuffers) X(glGenRenderbuffers) X(glBindRenderbuffer) \
    X(glRenderbufferStorage) X(glFramebufferRenderbuffer) X(glDeleteRenderbuffers)

#define PRT_DECLARE(name) extern PFN_##name name;
PRT_GL_FUNCS(PRT_DECLARE)
#undef PRT_DECLARE

// Load all pointers; requires a current GL context. Returns false if any is missing.
bool prt_gl_load();
