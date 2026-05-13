import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { motion, AnimatePresence } from "framer-motion";
import { Send, GraduationCap, RotateCcw } from "lucide-react";

interface Question {
  id: number;
  question: string;
  reference_answer: string;
  keywords: string[];
}

const GradeDisplay = ({ grade }: { grade: number }) => {
  const color =
    grade >= 6
      ? "text-success"
      : grade >= 4
      ? "text-yellow-500"
      : "text-destructive";

  return (
    <div className="flex flex-col items-center gap-2 animate-fade-in">
      <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
        Sua Nota
      </span>
      <span className={`text-8xl font-bold ${color}`}>{grade}</span>
      <span className="text-muted-foreground text-sm">/10</span>
    </div>
  );
};

const Index = () => {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [isLoadingQuestions, setIsLoadingQuestions] = useState(true);
  const [question, setQuestion] = useState<Question | null>(null);
  const [answer, setAnswer] = useState("");
  const [grade, setGrade] = useState<number | null>(null);
  const [gradeLabel, setGradeLabel] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Buscar questões da API ao carregar
  useEffect(() => {
    const fetchQuestions = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/v1/questions');
        if (!response.ok) throw new Error('Failed to fetch questions');
        
        const data = await response.json();
        const questionsArray = data.data?.questions || [];
        setQuestions(questionsArray);
        
        // Seleciona especificamente a questão ID 6
        if (questionsArray.length > 0) {
          const q6 = questionsArray.find((q: Question) => q.id === 6);
          if (q6) {
            setQuestion(q6);
          } else {
            // Fallback para a última se ID 6 não existir (segurança)
            setQuestion(questionsArray.at(-1));
          }
        }
      } catch (error) {
        console.error('Error fetching questions:', error);
        alert('Erro ao carregar as questões. Verifique se a API está rodando em http://localhost:8000');
      } finally {
        setIsLoadingQuestions(false);
      }
    };

    fetchQuestions();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!answer.trim() || isSubmitting || !question) return;

    setIsSubmitting(true);

    try {
      const response = await fetch(`http://127.0.0.1:8000/api/v1/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          student_answer: answer,
          question_id: question.id,
        }),
      });

      if (!response.ok) throw new Error("Failed to grade");
      
      const data = await response.json();
      console.log("API Response:", data);
      
      const score = data.data?.score || 0;
      const label = data.data?.feedback || "";
      setGrade(score);
      setGradeLabel(label);
      
    } catch (error) {
      console.error("Grading error:", error);
      alert("Erro ao avaliar a resposta. Verifique se a API está rodando em http://localhost:8000");
      setGrade(0);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    setAnswer("");
    setGrade(null);
    setGradeLabel(null);
    
    // Mantém a questão ID 6 ao resetar
    if (questions.length > 0) {
      const q6 = questions.find((q: Question) => q.id === 6);
      if (q6) {
        setQuestion(q6);
      }
    }
  };

  if (isLoadingQuestions) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-background">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
          className="h-8 w-8 rounded-full border-4 border-primary border-t-transparent"
        />
        <p className="mt-4 text-muted-foreground">Carregando questões...</p>
      </div>
    );
  }

  if (!question) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-background px-4">
        <p className="text-center text-lg text-muted-foreground">Nenhuma questão disponível</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-primary py-4 px-6 shadow-md">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <GraduationCap className="h-8 w-8 text-primary-foreground" />
          <h1 className="text-xl font-bold text-primary-foreground tracking-wide">
            QuestIA
          </h1>
        </div>
      </header>

      {/* Blue accent bar */}
      <div className="h-1 bg-secondary" />

      {/* Main */}
      <main className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-2xl">
          <div className="bg-card rounded-lg shadow-lg overflow-hidden border-t-4 border-t-secondary">
            {/* Card header */}
            <div className="bg-primary px-6 py-4">
              <h2 className="text-primary-foreground font-semibold text-lg">
                Avaliação
              </h2>
            </div>

            <div className="p-6 space-y-6">
              {/* Question */}
              <div className="space-y-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Questão
                </span>
                <p className="text-foreground text-lg font-medium leading-relaxed">
                  {question.question}
                </p>
              </div>

              {grade === null ? (
                <>
                  {/* Answer area */}
                  <div className="space-y-2">
                    <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Sua Resposta
                    </span>
                    <Textarea
                      value={answer}
                      onChange={(e) => setAnswer(e.target.value)}
                      placeholder="Digite sua resposta aqui..."
                      className="min-h-[150px] resize-none text-base focus-visible:ring-secondary"
                      disabled={isSubmitting}
                    />
                  </div>

                  {/* Submit */}
                  <div className="flex justify-end">
                    <Button
                      onClick={handleSubmit}
                      disabled={!answer.trim() || isSubmitting}
                      className="bg-secondary text-secondary-foreground hover:bg-secondary/90 gap-2 px-6"
                    >
                      {isSubmitting ? (
                        <div className="h-4 w-4 border-2 border-secondary-foreground/30 border-t-secondary-foreground rounded-full animate-spin" />
                      ) : (
                        <Send className="h-4 w-4" />
                      )}
                      {isSubmitting ? "Corrigindo..." : "Enviar Resposta"}
                    </Button>
                  </div>
                </>
              ) : (
                /* Grade result */
                <div className="flex flex-col items-center gap-6 py-8">
                  <GradeDisplay grade={grade} />
                  <Button
                    onClick={handleReset}
                    variant="outline"
                    className="gap-2 border-secondary text-secondary hover:bg-secondary/10"
                  >
                    <RotateCcw className="h-4 w-4" />
                    Tentar Novamente
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-primary py-3 text-center">
        <p className="text-primary-foreground/60 text-sm">
          QuestIA © 2026
        </p>
      </footer>
    </div>
  );
};

export default Index;
